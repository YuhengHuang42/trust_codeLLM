# Lookback Lens: Detecting and Mitigating Contextual Hallucinations in Large Language Models Using Only Attention Maps

import torch
import os
import typer
from typing_extensions import Annotated
from pathlib import Path
import json
import yaml
from loguru import logger
import tqdm
import numpy as np

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
from method.detect_model import LBLRegression, EncoderClassifier, collect_attention_map, collect_hidden_states
from method.extract import extract_util

app = typer.Typer(pretty_exceptions_show_locals=False, pretty_exceptions_short=False)

def get_attn_snapshot(model, tokenizer, data, cleaner=None):
    result = dict()
    for key in tqdm.tqdm(data):
        #str_input = data[key]['str_all']
        #if cleaner is not None:
        #    str_input = cleaner.forward(str_input)
        #tokenized_info = tokenizer(str_input, return_tensors="pt")
        tokenized_info = {"input_ids": torch.tensor(data[key]['token_output']).reshape(1, -1), 
                          "attention_mask": torch.tensor([1 for i in range(len(data[key]['token_output']))]).reshape(1, -1)
                          }
        with torch.inference_mode():
            snapshot = model.forward(**tokenized_info, output_attentions=True)
            attn_snapshot = [i.cpu() for i in snapshot['attentions']]
        result[key] = attn_snapshot # [layer_num, num_heads, seq_len, seq_len]
    return result

def get_hidden_snapshot(model, layer, data):
    recorder = extract_util.TransHookRecorder({layer: {"return_first": True}}, model, mode="plain")
    result = dict()
    for key in tqdm.tqdm(data):
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
            

'''

'''
@app.command()
def main(
    important_label_path: Annotated[Path, typer.Option()],
    data_path: Annotated[Path, typer.Option()],
    config_file: Annotated[Path, typer.Option()],
    model_save_path: Annotated[Path, typer.Option()],
    training_data_folder: Annotated[Path, typer.Option()] = None,
    parallel: Annotated[bool, typer.Option("--parallel/--no-parallel")] = True,
):
    with open(config_file, 'r') as file:
        config_dict = yaml.safe_load(file)
    cache_dir = None
    if "HF_HOME" in config_dict["system_setting"]:
        os.environ["HF_HOME"] = config_dict["system_setting"]["HF_HOME"]
    if "cache_dir" in config_dict["system_setting"]:
        cache_dir = config_dict["system_setting"]["cache_dir"]
    else:
        cache_dir = None
    
    with open(important_label_path, "r") as ifile:
        error_line_info = json.load(ifile)
    data = utils.load_shelve(data_path)
    
    model_name = config_dict["llm_config"]["model_name"]
    layer = config_dict["task_config"]["layer"]
    training_data_file_name = config_dict["task_config"]["training_data_file_name"]
    fit_model_param = config_dict["task_config"]["fit_model_param"]
    encoder_path = config_dict['task_config'].get("encoder_path", None)
    collect_type = config_dict["task_config"]['collect_type']
    mode = config_dict["task_config"]['mode']
    assert collect_type in ["attention", "hidden"]
    assert mode in ["lbl", "sae"]
    if mode == "sae":
        assert encoder_path is not None
        encoder_param = torch.load(encoder_path)
        encoder = sae_model.Autoencoder.from_state_dict(encoder_param["model_state_dict"])
    
    ## Load model
    quantization = config_dict["llm_config"]["quantization"]
    #generate_config = config_dict["llm_config"]["generate_config"]
    tokenizer = utils.load_tokenizer(model_name)

    target_buggy_positions, important_token_info = utils.get_important_token_pos(error_line_info, data, tokenizer)
    
    training_data_path = os.path.join(training_data_folder, training_data_file_name.format(layer))
    if os.path.exists(training_data_path):
        logger.info(f"Load {collect_type} training data from {training_data_path}")
        data_collection_flag = True
    else:
        logger.info(f"{collect_type} Training data not found at {training_data_path}")
        logger.info("Begin collecting data")
        data_collection_flag = False
        model, tokenizer = utils.load_opensource_model(model_name, parallel=parallel, quantization=quantization, cache_dir=cache_dir)
    
    train_x = []
    train_y = []
    
    if data_collection_flag:
        training_data = torch.load(training_data_path)
        train_x = training_data["train_x"]
        train_y = training_data["train_y"]
    else:
        #cleaner = utils.ModelOutputCleaner(start_token, model_name)
        if collect_type == "attention":
            snap_shot = get_attn_snapshot(model, tokenizer, data, None)
        elif collect_type == "hidden":
            snap_shot = get_hidden_snapshot(model, layer, data)
        label_dict = dict()
        for key in data:
            candidate_tokens = dataset_utils.get_candidate_tokens(data, key, tokenizer, "python", code_blocks_info=[[0, len(data[key]["str_output"])]]) # This is only for HumanEval
            if key in important_token_info:
                labels = utils.label_seg(important_token_info[key], candidate_tokens)
            else:
                labels = [0 for i in range(len(candidate_tokens))]
            label_dict[key] = labels
            snap_shot_single = snap_shot[key]
            input_token_length = data[key]["input_length"]
            if mode == "lbl":
                al_result = collect_attention_map(snap_shot_single, layer, input_token_length, candidate_tokens)
                train_x += [i for i in al_result]
            elif mode == "sae":
                latent_activations = collect_hidden_states(snap_shot_single, input_token_length, candidate_tokens, encoder).numpy()
                train_x += [i for i in latent_activations]
            train_y += label_dict[key]
        torch.save({"train_x": train_x, "train_y": train_y}, training_data_path)
    
    #for key in data:
    #    input_token_length = data[key]["input_length"]
    #    attn_snapshot = attn_data[key]
    #    al_result = collect_attention_map(attn_snapshot, attn_layer, input_token_length, candidate_tokens)
    #    train_x += [i for i in al_result]
    #    train_y += label_dict[key]
    train_x = np.stack(train_x)
    
    if mode == "lbl":
        clf = LBLRegression()
        clf.fit(train_x, train_y, "svm", fit_model_param, layer)
    elif mode == "sae":
        clf = EncoderClassifier()
        clf.fit(train_x, train_y, "svm", fit_model_param, encoder)
    clf.save(model_save_path)
    
    
if __name__ == "__main__":
    app()
    