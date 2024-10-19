# Lookback Lens: Detecting and Mitigating Contextual Hallucinations in Large Language Models Using Only Attention Maps

import torch
import os
import typer
from typing_extensions import Annotated
from typing import List
from pathlib import Path
import json
import yaml
from loguru import logger
import tqdm
import numpy as np
import pandas as pd
import time
import gc

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
# Add the project root to sys.path
sys.path.append(str(project_root))

import utility.utils as utils
import method.sae_model as sae_model
from task import dataset_utils
from task import safim
from method.detect_model import LBLRegression, EncoderClassifier, collect_attention_map, collect_hidden_states
from method.extract import extract_util

app = typer.Typer(pretty_exceptions_show_locals=False, pretty_exceptions_short=False)

LAYER_DICT = {
    "Phind/Phind-CodeLlama-34B-v2": {
        -1: "model.layers.47.self_attn"
    }
}
class TracerData():
    def __init__(self, 
                 data_path, 
                 tokenizer,
                 additional_prompt="", 
                 random_sample=None, 
                 max_length=2048):
        self.data_path = data_path
        self.raw_data = pd.read_csv(data_path, index_col=0)
        self.random_sample = random_sample
        if random_sample is not None:
            self.raw_data = self.raw_data.sample(random_sample) 
        self.error_code = self.raw_data['sourceText']
        self.correct_code = self.raw_data["targetText"]
        self.error_linenum = [max(i - 1, len(self.error_code[idx].splitlines()) - 1) for idx, i in enumerate(self.raw_data['lineNums_Abs'])] # From 1-based to 0-based
        self.error_info = self.raw_data['sourceErrorClangParse']
        self.tokenizer = tokenizer
        self.token_output = []
        max_length = min(max_length, tokenizer.model_max_length)
        if len(additional_prompt) > 0:
            inputs = tokenizer(additional_prompt, return_tensors="pt")
            self.input_length = inputs["input_ids"].shape[-1]
        else:
            self.input_length = 0
        drop_list = []
        for idx in range(len(self.error_code)):
            input_str = additional_prompt + self.error_code[idx]
            input_ids = tokenizer(input_str, return_tensors="pt", truncation=True, max_length=max_length)['input_ids']
            if input_ids.shape[-1] >= max_length:
                drop_list.append(idx)
            self.token_output.append(input_ids)

        self.error_code = [value for i, value in enumerate(self.error_code) if i not in drop_list]
        self.correct_code = [value for i, value in enumerate(self.correct_code) if i not in drop_list]
        self.error_linenum = [value for i, value in enumerate(self.error_linenum) if i not in drop_list]
        self.error_info = [value for i, value in enumerate(self.error_info) if i not in drop_list]
        self.token_output = [value for i, value in enumerate(self.token_output) if i not in drop_list]
        self.raw_data = self.raw_data.drop(index=drop_list).reset_index(drop=True)

    def __getitem__(self, idx):
        return_data = {
            "str_output": self.error_code[idx],
            "correct_code": self.correct_code[idx],
            "error_linenum": self.error_linenum[idx],
            "error_info": self.error_info[idx],
            "token_output": self.token_output[idx],
            "input_length": self.input_length
        }
        return return_data
    
    def __len__(self):
        return len(self.raw_data)
    
    def get_important_line_info(self):
        result = dict()
        for idx in range(len(self.error_linenum)):
            result[idx] = [
                [self.error_linenum[idx]]
            ]
        return result
    
    def keys(self):
        return [i for i in range(len(self.error_code))]
    
def get_attn_snapshot(model, tokenizer, data, layer, cleaner=None, keys=None):
    #assert layer == -1
    recorder = extract_util.TransHookRecorder({layer: {"output_attentions": True}}, model)
    result = dict()
    oom_keys = list()
    if keys is None:
        keys = data.keys()
    for key in tqdm.tqdm(keys):
        #str_input = data[key]['str_all']
        #if cleaner is not None:
        #    str_input = cleaner.forward(str_input)
        #tokenized_info = tokenizer(str_input, return_tensors="pt")
        tokenized_info = {"input_ids": torch.tensor(data[key]['token_output']).reshape(1, -1), 
                          "attention_mask": torch.tensor([1 for i in range(len(data[key]['token_output']))]).reshape(1, -1)
                          }
        with torch.inference_mode():
            try:
                past_key_values = extract_util.OffloadedCache() # GPU Memory Efficient
                tokenized_info["past_key_values"] = past_key_values
                _, record_dict = recorder.forward(tokenized_info)
                attn_snapshot = record_dict[list(recorder.layer_names.keys())[0]] # Only one layer
                #snapshot = model.forward(**tokenized_info, output_attentions=True)
                #attn_snapshot = [i.cpu() for i in snapshot['attentions']] # [layer_num, num_heads, seq_len, seq_len]
            except torch.cuda.OutOfMemoryError as e:
                oom_keys.append(key)
                torch.cuda.empty_cache()
                logger.info(f"Out of Memory error occurred for key: {key}. Moving to next.")
                continue
        result[key] = [attn_snapshot] # FIXME for layer != -1
    return result, oom_keys

def get_hidden_snapshot(model, layer, data):
    recorder = extract_util.TransHookRecorder({layer: {"return_first": True}}, model, mode="plain")
    result = dict()
    for key in tqdm.tqdm(data.keys()):
        tokenized_info = {"input_ids": torch.tensor(data[key]['token_output']).reshape(1, -1), 
                          "attention_mask": torch.tensor([1 for i in range(len(data[key]['token_output']))]).reshape(1, -1)
                          }
        with torch.inference_mode():
            _, record_dict = recorder.forward(tokenized_info)
            hidden_states = record_dict[list(recorder.layer_names.keys())[0]] # Only one layer
            hidden_states = hidden_states[0][0] # (token_num, hidden_size)
            recorder.clear_cache()
        result[key] = hidden_states
    return result
            
def clean_data(shelve_data):
    for key in shelve_data.keys():
        if len(shelve_data[key]['gen_probs'][0]) <= 1:
            shelve_data.pop(key)
    return shelve_data

def get_error_line_info_from_completion(shelve_data, upper_bound_abs_num=5):
    error_line_info = dict()
    pop_key = []
    dataset = safim.SAFIMDataset()
    for key in shelve_data:
        if shelve_data[key]['code_correctness'] != "success":
            # We assume the first line is the error line in code completion dataset.
            gt = dataset[int(key)]['ground_truth']
            gt_lines = len(gt.splitlines())
            if len(shelve_data[key]['line_number']) == 0:
                pop_key.append(key)
                continue
            labelled_line_num = shelve_data[key]['line_number'][0]
            if gt_lines + upper_bound_abs_num < len(labelled_line_num):
                pop_key.append(key)
                continue
            error_line_info[key] = [labelled_line_num]
    return error_line_info, pop_key
    
'''

'''
@app.command()
def main(
    #important_label_path: Annotated[List[str], typer.Option()],
    #data_path: Annotated[Path, typer.Option()],
    config_file: Annotated[Path, typer.Option()],
    model_save_path: Annotated[Path, typer.Option()],
    training_data_folder: Annotated[Path, typer.Option()] = None,
    parallel: Annotated[bool, typer.Option("--parallel/--no-parallel")] = True,
    encoder_path: Annotated[Path, typer.Option()] = None
):
    FALLBACK_ARGS = {
        "quantization": "4bit",
        "user_args": {"attn_implementation":"eager"},
        "device_map": "balanced"
    }
    with open(config_file, 'r') as file:
        config_dict = yaml.safe_load(file)
    cache_dir = None
    if "HF_HOME" in config_dict["system_setting"]:
        os.environ["HF_HOME"] = config_dict["system_setting"]["HF_HOME"]
    if "cache_dir" in config_dict["system_setting"]:
        cache_dir = config_dict["system_setting"]["cache_dir"]
    else:
        cache_dir = None
    model_name = config_dict["llm_config"]["model_name"]
    layer = config_dict["task_config"]["layer"]
    training_data_file_name = config_dict["task_config"]["training_data_file_name"]
    fit_model_param = config_dict["task_config"].get("fit_model_param", {})
    if encoder_path is None:
        encoder_path = config_dict['task_config'].get("encoder_path", None)
    collect_type = config_dict["task_config"]['collect_type']
    mode = config_dict["task_config"]['mode']
    random_sample = config_dict["task_config"].get("random_sample", None)
    additional_prompt = config_dict["task_config"].get("additional_prompt", "")
    dataset_list = config_dict["task_config"].get("dataset_list", ["humaneval"])
    data_source = config_dict["task_config"].get("data_source", ["shelve"])
    important_label_path_list = config_dict["task_config"].get("important_label_path_list", [None])
    data_path_list = config_dict["task_config"]['data_path_list']
    #assert data_source in ["shelve", "tracer"]
    assert collect_type in ["attention", "hidden"]
    assert mode in ["lbl", "sae"]
    #assert dataset in ["humaneval", "safim"]
    FALLBACK_ARGS["model_name"] = model_name
    FALLBACK_ARGS["parallel"] = parallel
    FALLBACK_ARGS["cache_dir"] = cache_dir
    
    if mode == "sae":
        assert encoder_path is not None
        encoder_param = torch.load(encoder_path)
        encoder = sae_model.Autoencoder.from_state_dict(encoder_param["model_state_dict"])

    tokenizer = utils.load_tokenizer(model_name)
    data_line_token_pair = [[], [], []]
    for idx, source in enumerate(data_source):
        important_label_path = important_label_path_list[idx]
        data_path = data_path_list[idx]
        dataset = dataset_list[idx]
        if source == "shelve":
            data = utils.load_shelve(data_path)
            data = clean_data(data)
            if dataset == "safim":
                error_line_info, pop_key = get_error_line_info_from_completion(data)
                for key in pop_key:
                    data.pop(key)
            else:
                with open(important_label_path, "r") as ifile:
                    error_line_info = json.load(ifile)
        elif source == "tracer":
            data = TracerData(data_path, tokenizer, random_sample=random_sample, additional_prompt=additional_prompt)
            error_line_info = data.get_important_line_info()
        data_line_token_pair[0].append(data)
        data_line_token_pair[1].append(error_line_info)
        target_buggy_positions, important_token_info = utils.get_important_token_pos(error_line_info, data, tokenizer)
        data_line_token_pair[2].append(important_token_info)
        
    ## Load model
    quantization = config_dict["llm_config"]["quantization"]
    #generate_config = config_dict["llm_config"]["generate_config"]
    
    training_data_path = os.path.join(training_data_folder, training_data_file_name.format(layer))
    if os.path.exists(training_data_path):
        #logger.info(f"Load {collect_type} training data from {training_data_path}")
        data_collection_flag = True
    else:
        logger.info(f"{collect_type} Training data not found at {training_data_path}")
        logger.info("Begin collecting data")
        data_collection_flag = False
        model, tokenizer = utils.load_opensource_model(model_name, parallel=parallel, quantization=quantization, cache_dir=cache_dir)
    
    train_x = []
    snapshot_x = {}
    store_y = {}
    train_y = []
    
    if data_collection_flag:
        training_data = torch.load(training_data_path)
        logger.info(f"Load {collect_type} training data from {training_data_path}")
        snapshot_x = training_data["snapshot_x"]
        store_y = training_data["store_y"]
        for idx, dataset_name in enumerate(dataset_list):
            for key in snapshot_x[dataset_name]:
                snap_shot_single = snapshot_x[dataset_name][key]
                data = data_line_token_pair[0][idx]
                candidate_tokens = dataset_utils.get_candidate_tokens(data, key, tokenizer, "python", code_blocks_info=[[0, len(data[key]["str_output"])]]) # This is only for HumanEval
                input_token_length = data[key]["input_length"]
                if mode == "lbl":
                    al_result = collect_attention_map(snap_shot_single, layer, input_token_length, candidate_tokens)
                    train_x += [i for i in al_result]
                elif mode == "sae":
                    latent_activations = collect_hidden_states(snap_shot_single, input_token_length, candidate_tokens, encoder).numpy()
                    train_x += [i for i in latent_activations]
                train_y += store_y[dataset_name][key]
    else:
        for idx in range(len(data_line_token_pair[0])):
            #cleaner = utils.ModelOutputCleaner(start_token, model_name)
            data = data_line_token_pair[0][idx]
            error_line_info = data_line_token_pair[1][idx]
            important_token_info = data_line_token_pair[2][idx]
            data_class_name = dataset_list[idx]
            snapshot_x[data_class_name] = dict()
            store_y[data_class_name] = dict()
            if collect_type == "attention":
                layer_name = LAYER_DICT[model_name][layer]
                snap_shot_first, oom_keys = get_attn_snapshot(model, tokenizer, data, layer_name, None)
                snap_shot_second = dict()
                if len(oom_keys) > 0:
                    logger.info("Begin fallback inference for OOM data points")
                    del recorder
                    del model
                    time.sleep(3)
                    gc.collect()
                    torch.cuda.empty_cache()
                    model, tokenizer = utils.load_opensource_model(**FALLBACK_ARGS)
                    snap_shot_second, oom_keys = get_attn_snapshot(model, tokenizer, data, layer_name, None, keys=oom_keys)
                    if len(oom_keys) > 0:
                        logger.error("OOM error still exists after fallback inference. Ignore them.")
                        logger.error("OOM keys: {}".format(oom_keys))
                snap_shot = {**snap_shot_first, **snap_shot_second}
            elif collect_type == "hidden":
                snap_shot = get_hidden_snapshot(model, layer, data)
            label_dict = dict()
            for key in data.keys():
                candidate_tokens = dataset_utils.get_candidate_tokens(data, key, tokenizer, "python", code_blocks_info=[[0, len(data[key]["str_output"])]]) # This is only for HumanEval
                if key in important_token_info:
                    labels = utils.label_seg(important_token_info[key], candidate_tokens)
                else:
                    labels = [0 for i in range(len(candidate_tokens))]
                label_dict[key] = labels
                snap_shot_single = snap_shot[key]
                snapshot_x[data_class_name][key] = snap_shot_single
                input_token_length = data[key]["input_length"]
                if mode == "lbl":
                    al_result = collect_attention_map(snap_shot_single, layer, input_token_length, candidate_tokens)
                    train_x += [i for i in al_result]
                elif mode == "sae":
                    latent_activations = collect_hidden_states(snap_shot_single, input_token_length, candidate_tokens, encoder).numpy()
                    train_x += [i for i in latent_activations]
                train_y += label_dict[key]
                store_y[data_class_name][key] = label_dict[key]
        torch.save({"snapshot_x": snapshot_x, "store_y": store_y}, training_data_path)
    
    #for key in data:
    #    input_token_length = data[key]["input_length"]
    #    attn_snapshot = attn_data[key]
    #    al_result = collect_attention_map(attn_snapshot, attn_layer, input_token_length, candidate_tokens)
    #    train_x += [i for i in al_result]
    #    train_y += label_dict[key]
    train_x = np.stack(train_x)
    logger.info(f"Train X shape: {train_x.shape}")
    
    if mode == "lbl":
        clf = LBLRegression()
        clf.fit(train_x, train_y, "mlp", fit_model_param, layer)
    elif mode == "sae":
        clf = EncoderClassifier()
        clf.fit(train_x, train_y, "mlp", fit_model_param, encoder)
    clf.save(model_save_path)
    
    
if __name__ == "__main__":
    app()
    