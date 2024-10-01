import os

import torch
import tqdm
import yaml
import numpy as np
import tqdm
import typer
from loguru import logger
from typing_extensions import Annotated
from pathlib import Path
import json
import gc
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

import sys
project_root = Path(__file__).resolve().parent.parent
# Add the project root to sys.path
sys.path.append(str(project_root))

from method import lbl
import method.extract as extract
import utility.utils as utils
import task.defect4j as defects4j
from task import dataset_utils

app = typer.Typer(pretty_exceptions_show_locals=False, pretty_exceptions_short=False)

def compute_hit(important_tokens: list, selected_tokens: list):
    """
    Compute hit rate given the ground truth tokens and selected tokens.
    ---
    Args:
        important_tokens: List[List]. important_token_info[key] returned by utils.get_important_token_pos
        selected_tokens: List
    Return:
        Hit rate
    """
    gt = []
    for item in important_tokens:
        gt += item
    gt = set(gt)
    selected = set(selected_tokens)
    return len(selected.intersection(gt)) / len(gt) 

def evaluate_lbl(data, 
                 key_list, 
                 important_token_info, 
                 recorder, 
                 tokenizer,
                 lbl_model, 
                 language, 
                 attn_layer,
                 score=0,
                 counter=0,
                 topk=5,
                 max_profile_token_length=4096):
    oom_keys = []
    result = dict()
    for key in tqdm.tqdm(key_list):
        candidate_tokens = dataset_utils.get_candidate_tokens(data, key, tokenizer, language, code_blocks_info=[[0, len(data[key]["str_output"])]])
        token_length = len(data[key]['token_output'])
        tail_truncate = 0
        input_token_length = data[key]['input_length']
        if token_length > max_profile_token_length:
            tail_truncate = token_length - max_profile_token_length
            input_token_length = max(input_token_length - tail_truncate, 0)
            
        tokenized_info = {"input_ids": torch.tensor(data[key]['token_output']).reshape(1, -1)[:, tail_truncate:], 
                        "attention_mask": torch.tensor([1 for i in range(len(data[key]['token_output']))]).reshape(1, -1)
                        }
        with torch.inference_mode():
            # If transformer.__version__ == v4.45.1, directly import it from transformers.cache_utils 
            past_key_values = extract.OffloadedCache() # GPU Memory Efficient
            tokenized_info["past_key_values"] = past_key_values
            try:
                #snapshot = model.forward(**tokenized_info, output_attentions=True)
                snapshot, attn_ext = recorder.forward(tokenized_info)
            except torch.cuda.OutOfMemoryError as e:
                oom_keys.append(key)
                torch.cuda.empty_cache()
                logger.info(f"Out of Memory error occurred for key: {key}. Moving to next.")
                continue
        attn_snapshot = attn_ext[attn_layer]
        del snapshot
        # Inference process of LBL baseline method
        pred_result = lbl_model.predict([attn_snapshot], input_token_length, candidate_tokens)
        pred_result = pred_result[:, 1]
        result[key] = pred_result
        rank_per_line = sorted(list(zip(pred_result, [i for i in range(len(pred_result))])), reverse=True)[:topk]
        selected_token = set()
        for line in rank_per_line:
            selected_token = selected_token.union(set(candidate_tokens[line[1]]))
        
        topk_score = compute_hit(important_token_info[key], selected_token)
        score += topk_score
        counter += 1
    return score, counter, result, oom_keys
    
@app.command()
def main(
    important_label_path: Annotated[Path, typer.Option()],
    data_path: Annotated[Path, typer.Option()],
    config_file: Annotated[Path, typer.Option()],
    result_output_path: Annotated[Path, typer.Option()],
    parallel: Annotated[bool, typer.Option("--parallel/--no-parallel")] = True,
):
    FALLBACK_ARGS = {
        "quantization": "4bit",
        "user_args": {"attn_implementation":"eager"}
    }
    # ===== Load configuration =====
    with open(config_file, 'r') as file:
        config_dict = yaml.safe_load(file)
    model_name = config_dict["llm_config"]["model_name"]
    quantization = config_dict["llm_config"]["quantization"]
    attn_layer = config_dict["task_config"]["attn_layer"]
    language = config_dict["task_config"]["language"]
    lbl_model_path = config_dict["task_config"]["lbl_model_path"]
    max_profile_token_length = config_dict["task_config"]["max_profile_token_length"]
    cache_dir = None
    if "HF_HOME" in config_dict["system_setting"]:
        os.environ["HF_HOME"] = config_dict["system_setting"]["HF_HOME"]
    if "cache_dir" in config_dict["system_setting"]:
        cache_dir = config_dict["system_setting"]["cache_dir"]
    else:
        cache_dir = None
    # ===== Load Model =====
    model, tokenizer = utils.load_opensource_model(model_name, parallel=parallel, quantization=quantization, cache_dir=cache_dir)
    recorder = extract.TransHookRecorder({attn_layer: {"output_attentions": True}}, model)
    FALLBACK_ARGS["model_name"] = model_name
    FALLBACK_ARGS["parallel"] = parallel
    FALLBACK_ARGS["cache_dir"] = cache_dir

    # ===== Load Data =====
    data = utils.load_shelve(data_path)
    
    #if task.lower() == "defects4j":
    #    data_path = config_dict["task_config"]["repair_data_path"]
    #    loc_folder = config_dict["task_config"]["repair_loc_folder"]
    #    defects4j_path = config_dict["system_setting"]['DEFECTS4J_PATH']
    #    java_path = config_dict["system_setting"]['JAVA_PATH']
    #    dataset = defects4j.Defects4jDataset(data_path, loc_folder, defects4j_path, java_home=java_path)

    # ===== Obtain task specific data info =====
    with open(important_label_path, "r") as ifile:
        error_line_info = json.load(ifile)
    #error_line_info = {}
    #for key in data:
    #    diff_results = utils.get_changes_with_line_numbers(dataset[int(key)]['fix'], data[key]['str_output'], "java")
    #    error_line_info[key] =  [list(set([i[0] for i in diff_results[0]] + [i[0] for i in diff_results[1]]))]
    target_buggy_positions, important_token_info = utils.get_important_token_pos(error_line_info, data, tokenizer)
    evaluate_key_list = []
    for key in data:
        if data[key]["code_correctness"] == 'correct':
            continue
        if len(error_line_info[key][0]) == 0:
            logger.info("No error info for key: {}".format(key))
            continue
        evaluate_key_list.append(key)

    # ==== Begin collecting attention scores ====
    lbl_model = lbl.LBLRegression()
    lbl_model.load(lbl_model_path)
    
    score, counter, first_result, oom_keys = evaluate_lbl(data, 
                                                        evaluate_key_list, 
                                                        important_token_info, 
                                                        recorder, 
                                                        tokenizer,
                                                        lbl_model, 
                                                        language, 
                                                        attn_layer,
                                                        max_profile_token_length=max_profile_token_length
                                                        )
    
    
    if len(oom_keys) > 0:
        logger.info("Begin fallback inference for OOM data points")
        del recorder
        del model
        gc.collect()
        torch.cuda.empty_cache()
        model, tokenizer = utils.load_opensource_model(**FALLBACK_ARGS)
        recorder = extract.TransHookRecorder({attn_layer: {"output_attentions": True}}, model)
        score, counter, second_result, oom_keys = evaluate_lbl(data, 
                                                            oom_keys, 
                                                            important_token_info, 
                                                            recorder, 
                                                            tokenizer,
                                                            lbl_model, 
                                                            language, 
                                                            attn_layer,
                                                            score=score,
                                                            counter=counter,
                                                            max_profile_token_length=max_profile_token_length
                                                            )
        if len(oom_keys) > 0:
            logger.error("OOM error still exists after fallback inference. Ignore them.")
            logger.error("OOM keys: {}".format(oom_keys))
    
    topk_recorder = {"top5": score / counter}
    result = {**first_result, **second_result}
    
    for topk_iter in [1, 3, 10]:
        score = 0
        counter = 0
        for key in result:
            pred_result = result[key]
            rank_per_line = sorted(list(zip(pred_result, [i for i in range(len(pred_result))])), reverse=True)[:topk_iter]
            candidate_tokens = dataset_utils.get_candidate_tokens(data, key, tokenizer, language, code_blocks_info=[[0, len(data[key]["str_output"])]])
            selected_token = set()
            for line in rank_per_line:
                selected_token = selected_token.union(set(candidate_tokens[line[1]]))
            
            topk_score = compute_hit(important_token_info[key], selected_token)
            score += topk_score
            counter += 1
        topk_recorder["top{}".format(topk_iter)] = score / counter
    
    for key in result:
        result[key] = result[key].tolist()
    
    with open(result_output_path, "w") as ofile:
        json.dump({"result": topk_recorder, "pred_result": result, "OOM_keys": oom_keys}, ofile)

if __name__ == "__main__":
    app()