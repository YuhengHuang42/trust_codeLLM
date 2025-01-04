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
import method.detect_model as detect_model
from method.detect_model import LBLRegression, EncoderClassifier, collect_attention_map, collect_hidden_states, flat_data_dict
from method.extract import extract_util
from method.detect_model import aggregate_feature

app = typer.Typer(pretty_exceptions_show_locals=False, pretty_exceptions_short=False)

LAYER_DICT = {
    "Phind/Phind-CodeLlama-34B-v2": {
        -1: "model.layers.47.self_attn"
    }
}
from utility.utils import HARD_TOKEN_LIMIT

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
    all_keys = list(shelve_data.keys())
    for key in all_keys:
        if len(shelve_data[key]['gen_probs'][0]) <= 1:
            shelve_data.pop(key)
            continue
        if shelve_data[key]['input_length'] >= HARD_TOKEN_LIMIT:
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

    
def load_gt(data_source, dataset_list, data_path_list, important_label_path_list, tokenizer):
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
        error_line_info_keys = list(error_line_info.keys())
        for key in error_line_info_keys:
            if key not in data:
                error_line_info.pop(key)
        data_line_token_pair[0].append(data)
        data_line_token_pair[1].append(error_line_info)
        target_buggy_positions, important_token_info = utils.get_important_token_pos(error_line_info, data, tokenizer, language="python")
        data_line_token_pair[2].append(important_token_info)
    return data_line_token_pair

def load_from_existing_data(training_data_path,
                            mode, 
                            #pass_mode, 
                            encoder,
                           ):
    training_data = torch.load(training_data_path)
    train_x = []
    train_y = []
    snapshot_x = training_data["snapshot_x"]
    store_y = training_data["store_y"]
    first_token_info = training_data["first_token"]
    candidate_token_dict = training_data["candidate_token_dict"]
    #if pass_mode == "dict":
    train_x = {}
    train_y = {}
    for key in snapshot_x:
        #if key not in data:
        #    continue
        snap_shot_single = snapshot_x[key]
        first_token_single = first_token_info[0][key] if key in first_token_info[0] else None
        #first_token_dict = first_token_info[0][dataset_name][key]
        # This is only for HumanEval and SAFIM
        #candidate_tokens = dataset_utils.get_candidate_tokens(data, key, tokenizer, language, code_blocks_info=[[0, len(data[key]["str_output"])]]) 
        #input_token_length = data[key]["input_length"]
        if mode == "lbl":
            #al_result = collect_attention_map(snap_shot_single, layer, input_token_length, candidate_tokens)
            #if pass_mode == "dict":
            #train_x[key] = torch.stack(snap_shot_single)
            train_x[key] = snap_shot_single
            #else:
            #   train_x += [i for i in snap_shot_single]
        elif (mode == "sae" or mode =="ae") or mode == "internal_only":
            #latent_activations, _ = collect_hidden_states(snap_shot_single, input_token_length, candidate_tokens, encoder).numpy()
            with torch.inference_mode():
                snap_shot_single = torch.from_numpy(snap_shot_single)
                if (mode == "sae" or mode =="ae"):
                    latent_activations, info = encoder.encode(snap_shot_single.to(encoder.device))
                    first_act, info = encoder.encode(first_token_single.to(encoder.device))
                elif mode == "internal_only":
                    latent_activations = snap_shot_single
                    first_act = None
                #if pass_mode == "dict":
                train_x[key] = latent_activations.cpu()
                #else:
                #    train_x += [i for i in latent_activations]
                if first_act is not None:
                    first_token_info[0][key] = first_act.cpu()
                else:
                    first_token_info[0][key] = None
        #if pass_mode == "dict":
        train_y[key] = store_y[key]
        #else:
        #    train_y += store_y[key]
    return train_x, train_y, first_token_info[0], first_token_info[1], candidate_token_dict
    
@app.command()
def main(
    #important_label_path: Annotated[List[str], typer.Option()],
    #data_path: Annotated[Path, typer.Option()],
    config_file: Annotated[Path, typer.Option()],
    model_save_path: Annotated[Path, typer.Option()],
    training_data_folder: Annotated[Path, typer.Option()] = None,
    parallel: Annotated[bool, typer.Option("--parallel/--no-parallel")] = True,
    encoder_path: Annotated[Path, typer.Option()] = None,
    agg: Annotated[str, typer.Option()] = None, # agg is used to indicate the aggregation method for hidden states used in classification mode.
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
    extract_code_list = config_dict["task_config"]["extract_code_list"]
    training_data_path_name = config_dict["task_config"]["training_data_path_name"]
    fit_model_param = config_dict["task_config"].get("fit_model_param", {})
    if encoder_path is None:
        encoder_path = config_dict['task_config'].get("encoder_path", None)
    collect_type = config_dict["task_config"]['collect_type']
    mode = config_dict["task_config"]['mode']
    #random_sample = config_dict["task_config"].get("random_sample", None)
    #additional_prompt = config_dict["task_config"].get("additional_prompt", "")
    dataset_list = config_dict["task_config"].get("dataset_list", ["humaneval"])
    data_source = config_dict["task_config"].get("data_source", ["shelve"])
    important_label_path_list = config_dict["task_config"].get("important_label_path_list", [None])
    data_path_list = config_dict["task_config"]['data_path_list']
    vector_norm = config_dict["task_config"].get("vector_norm", False)
    model_type = config_dict["task_config"].get("model_type", "mlp")
    external_proj_path = config_dict["task_config"].get("external_proj_path", None)
    model_type = model_type.lower()
    logger.info(f"Using Model Type: {model_type}")
    if model_type == "lstm" or model_type == "attn":
        pass_mode = "dict"
    else:
        pass_mode = "list"
    #assert data_source in ["shelve", "tracer"]
    assert collect_type in ["attention", "hidden"]
    assert mode in ["lbl", "sae", "internal_only", "ae"]
    #assert dataset in ["humaneval", "safim"]
    FALLBACK_ARGS["model_name"] = model_name
    FALLBACK_ARGS["parallel"] = parallel
    FALLBACK_ARGS["cache_dir"] = cache_dir
    
    if mode == "sae":
        assert encoder_path is not None
        encoder_param = torch.load(encoder_path)
        encoder = sae_model.Autoencoder.from_state_dict(encoder_param["model_state_dict"])
    elif mode == "ae":
        assert encoder_path is not None
        encoder_param = torch.load(encoder_path)
        encoder = sae_model.NaiveAutoEncoder.from_state_dict(encoder_param["model_state_dict"])
    else:
        logger.info("No encoder is used.")
        encoder = None

    if agg is not None:
        logger.info("Use aggregation method: {}".format(agg))
        
    if external_proj_path is not None:
        external_proj = detect_model.SupervisedPrjection.load(external_proj_path)
        logger.info(f"Load external projection from {external_proj_path}")
    else:
        external_proj = None
    tokenizer = utils.load_tokenizer(model_name)
    data_line_token_pair = load_gt(data_source, dataset_list, data_path_list, important_label_path_list, tokenizer)
    ## Load model
    quantization = config_dict["llm_config"]["quantization"]
    #generate_config = config_dict["llm_config"]["generate_config"]
    
    training_data_path = os.path.join(training_data_folder, training_data_path_name.format(layer))
    os.makedirs(training_data_path, exist_ok=True)
    #if os.path.exists(training_data_path):
    #    #logger.info(f"Load {collect_type} training data from {training_data_path}")
    #    logger.info(f"{training_data_path} already existed")
    #    data_collection_flag = True
    #else:
    #    logger.info(f"{collect_type} Training data not found at {training_data_path}")
    #    #logger.info("Begin collecting data")
    #    data_collection_flag = False
    #    model, tokenizer = utils.load_opensource_model(model_name, parallel=parallel, quantization=quantization, cache_dir=cache_dir)
    data_collection_flag = False
    if pass_mode == "dict":
        train_x = {key: {} for key in dataset_list}
        train_y = {key: {} for key in dataset_list}
    else:
        train_x = []
        train_y = []
    snapshot_x = {}
    store_y = {}
    first_token_dict = {}
    first_token_in_dict = {}
    candidate_token_dict = {}
    for data_class_idx, data_class_name in enumerate(dataset_list):
        cur_data_path = os.path.join(training_data_path, data_class_name)
        if os.path.exists(cur_data_path):
            logger.info(f"Load {collect_type} training data from {cur_data_path}")
            cur_train_x, cur_train_y, cur_first_token_dict, cur_first_token_in_dict, cur_candidate_token_dict = load_from_existing_data(cur_data_path, 
                                                                                                                                        mode,
                                                                                                                                        #data_line_token_pair[data_class_idx], 
                                                                                                                                        encoder)
            #if agg is None:
                # Line Level prediction mode
                # Keeps the same with semantic_binding_rank.py
            #    iter_list = list(cur_train_y.keys())
            #    for key in iter_list:
                    # The entire code is correct
            #        if sum(cur_train_y[key]) == 0:
            #            cur_train_x.pop(key)
            #            cur_train_y.pop(key)
            #            cur_first_token_dict.pop(key)
            #            cur_candidate_token_dict.pop(key)
            if agg is not None:
                for key in cur_train_x:
                    cur_train_x[key] = aggregate_feature(cur_train_x[key], agg) # (L, hidden_dim) --> (hidden_dim)
                for key in cur_train_y:
                    cur_train_y[key] = [max(cur_train_y[key])] # Assume 1 means incorrect code.
            if pass_mode == "dict":
                train_x[data_class_name] = cur_train_x
                train_y[data_class_name] = cur_train_y
            else:
                list_cur_train_x = []
                for key in cur_train_x:
                    if agg is not None:
                        list_cur_train_x.append(cur_train_x[key])
                    else:
                        list_cur_train_x += [i for i in cur_train_x[key]]
                list_cur_train_y = []
                for key in cur_train_y:
                    list_cur_train_y += cur_train_y[key]
                train_x += list_cur_train_x
                train_y += list_cur_train_y
            first_token_dict[data_class_name] = cur_first_token_dict
            candidate_token_dict[data_class_name] = cur_candidate_token_dict
            first_token_in_dict[data_class_name] = cur_first_token_in_dict
        else:
            if data_collection_flag is False:
                model, tokenizer = utils.load_opensource_model(model_name, parallel=parallel, quantization=quantization, cache_dir=cache_dir)
                data_collection_flag = True
            #if pass_mode == "dict":
            #    train_x = {key: {} for key in dataset_list}
            #    train_y = {key: {} for key in dataset_list}
            #for idx in range(len(data_line_token_pair[0])):
                #cleaner = utils.ModelOutputCleaner(start_token, model_name)
            #    data = data_line_token_pair[0][idx]
            #    error_line_info = data_line_token_pair[1][idx]
            #    important_token_info = data_line_token_pair[2][idx]
            
            data = data_line_token_pair[0][data_class_idx]
            error_line_info = data_line_token_pair[1][data_class_idx]
            important_token_info = data_line_token_pair[2][data_class_idx]
            
            snapshot_x[data_class_name] = dict()
            candidate_token_dict[data_class_name] = dict()
            store_y[data_class_name] = dict()
            first_token_dict[data_class_name] = dict()
            first_token_in_dict[data_class_name] = dict()
            if collect_type == "attention":
                layer_name = LAYER_DICT[model_name][layer]
                snap_shot_first, oom_keys = get_attn_snapshot(model, tokenizer, data, layer_name, None)
                snap_shot_second = dict()
                if len(oom_keys) > 0:
                    logger.info("Begin fallback inference for OOM data points")
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
                extract_code = extract_code_list[data_class_idx]
                if extract_code:
                    code_blocks_info = None
                else:
                    code_blocks_info = [[0, len(data[key]["str_output"])]]
                candidate_tokens = dataset_utils.get_candidate_tokens(data, key, tokenizer, "python", code_blocks_info=code_blocks_info)
                if candidate_tokens is None:
                    continue
                if key in important_token_info:
                    labels = utils.label_seg(important_token_info[key], candidate_tokens)
                else:
                    labels = [0 for i in range(len(candidate_tokens))]
                label_dict[key] = labels
                snap_shot_single = snap_shot[key]
                #snapshot_x[data_class_name][key] = snap_shot_single
                input_token_length = data[key]["input_length"]
                candidate_token_dict[data_class_name][key] = candidate_tokens
                if mode == "lbl":
                    al_result = collect_attention_map(snap_shot_single, layer, input_token_length, candidate_tokens)
                    al_result = torch.stack(al_result)
                    snapshot_x[data_class_name][key] = al_result.tolist()
                    if pass_mode == "dict":
                        if agg is None:
                            train_x[data_class_name][key] = al_result
                        else:
                            train_x[data_class_name][key] = aggregate_feature(al_result, agg)
                    else:
                        if agg is None:
                            train_x += [i for i in al_result]
                        else:
                            train_x.append(aggregate_feature(al_result, agg))
                elif (mode == "sae" or mode =="ae") or mode == "internal_only":
                    _, before_latent_activations = collect_hidden_states(snap_shot_single, input_token_length, candidate_tokens, None)
                    first_token_dict[data_class_name][key] = snap_shot_single[input_token_length]
                    first_token_in_dict[data_class_name][key] = (candidate_tokens[0] == 0)
                    with torch.inference_mode():
                        if (mode == "sae" or mode =="ae"):
                            latent_activations, info = encoder.encode(before_latent_activations.to(encoder.device))
                        elif mode == "internal_only":
                            latent_activations = before_latent_activations
                    snapshot_x[data_class_name][key] = before_latent_activations.cpu().numpy()
                    if pass_mode == "dict":
                        if agg is None:
                            train_x[data_class_name][key] = latent_activations.cpu().numpy()
                        else:
                            train_x[data_class_name][key] = aggregate_feature(latent_activations.cpu().numpy(), agg)
                    else:
                        if agg is None:
                            train_x += [i for i in latent_activations.cpu().numpy()]
                        else:
                            train_x.append(aggregate_feature(latent_activations.cpu().numpy(), agg))
                if pass_mode == "dict":
                    if agg is None:
                        train_y[data_class_name][key] = label_dict[key]
                    else:
                        train_y[data_class_name][key] = max(label_dict[key])
                else:
                    if agg is None:
                        train_y += label_dict[key]
                    else:
                        train_y.append(max(label_dict[key]))
                store_y[data_class_name][key] = label_dict[key]
            torch.save(
                {"snapshot_x": snapshot_x[data_class_name], 
                "store_y": store_y[data_class_name], 
                "first_token": [first_token_dict[data_class_name], first_token_in_dict[data_class_name]],
                "candidate_token_dict": candidate_token_dict[data_class_name],
                }, cur_data_path)
    
    #torch.save({"snapshot_x": snapshot_x, 
    #            "store_y": store_y[data_class_name], 
    #            "first_token": [first_token_dict, first_token_in_dict],
    #            "candidate_token_dict": candidate_token_dict,
    #            }, training_data_path)
    
    #for key in data:
    #    input_token_length = data[key]["input_length"]
    #    attn_snapshot = attn_data[key]
    #    al_result = collect_attention_map(attn_snapshot, attn_layer, input_token_length, candidate_tokens)
    #    train_x += [i for i in al_result]
    #    train_y += label_dict[key]
    train_info = dict()
    if pass_mode == "list":
        train_x = np.stack(train_x)
        logger.info(f"Train X shape: {train_x.shape}")
        logger.info(f"Train Y Pos Label: {sum(train_y)}")
        train_info["train_x"] = train_x
        train_info["train_y"] = train_y
    elif pass_mode == "dict":
        train_x = flat_data_dict(train_x)
        train_y = flat_data_dict(train_y)
        first_token_dict = flat_data_dict(first_token_dict)
        logger.info(f"Train X shape: {sum([train_x[key].shape[0] for key in train_x])}")
        logger.info(f"Train Y Pos Label: {sum([sum(train_y[key]) for key in train_y])}")
        train_info["train_x"] = train_x
        train_info["train_y"] = train_y
        train_info["context"] = first_token_dict
    if mode == "lbl":
        clf = LBLRegression()
        clf.fit(train_info, model_type, fit_model_param, layer)
    elif (mode == "sae" or mode =="ae") or mode == "internal_only":
        clf = EncoderClassifier()
        clf.fit(train_info, model_type, fit_model_param, encoder, vector_norm=vector_norm, external_proj=external_proj, agg=agg)
    clf.save(model_save_path)
    #accuracy, pred_profiler = clf.evaluate(train_x, train_y)
    #logger.info(f"Accuracy on training data: {accuracy}")
    #torch.save(pred_profiler, "pred_label_recoreder.pt")
    
    
if __name__ == "__main__":
    app()
    