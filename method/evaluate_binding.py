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
import time
import copy
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
project_root = Path(__file__).resolve().parent.parent
# Add the project root to sys.path
sys.path.append(str(project_root))

import method.extract.extract_util as extract_util
from method import detect_model
from detect_model import AttentionModule, ScaledDotProductAttention, RankNet
import utility.utils as utils
from task import dataset_utils

STORE = 0
LOAD = 1

app = typer.Typer(pretty_exceptions_show_locals=False, pretty_exceptions_short=False)

def compute_hit_line(important_tokens: list, selected_tokens: list):
    """
    Compute hit rate given the ground truth tokens and selected tokens.
    ---
    Args:
        important_tokens: List[List]. important_token_info[key] returned by utils.get_important_token_pos
        selected_tokens: List
    Return:
        Hit rate
    """
    selected_tokens = set(selected_tokens)
    result = 0
    for line in important_tokens:
        line_hit = True
        for token in line:
            if token not in selected_tokens:
                line_hit = False
                break
        result += line_hit
    return result / len(important_tokens) 

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

def compute_recall(important_tokens: list, selected_tokens: list):
    """
    Compute recall rate given the ground truth tokens and selected tokens.
    ---
    Args:
        important_tokens: List[List]. important_token_info[key] returned by utils.get_important_token_pos
        selected_tokens: List
    Return:
        Recall rate
    """
    gt = []
    for item in important_tokens:
        gt += item
    gt = set(gt)
    selected = set(selected_tokens)
    return len(selected.intersection(gt)) / len(selected)

def evaluate_binding(data, 
                 key_list, 
                 important_token_info, 
                 recorder, 
                 tokenizer,
                 detection_model, 
                 language, 
                 layer,
                 score=0,
                 counter=0,
                 topk=5,
                 max_profile_token_length=4096,
                 extract_code=True,
                 before_act_dict=None,
                 first_token_info_dict=None
                 ):
    oom_keys = []
    result = dict()
    #pred_profiler = list()
    if before_act_dict is None:
        mode = STORE
        before_act_dict = dict()
        first_token_info_dict = dict()
    else:
        mode = LOAD
    for key in tqdm.tqdm(key_list):
        if mode == STORE:
            if extract_code:
                code_blocks_info = None
            else:
                code_blocks_info = [[0, len(data[key]["str_output"])]]
            candidate_tokens = dataset_utils.get_candidate_tokens(data, key, tokenizer, language, code_blocks_info=code_blocks_info)
            token_length = len(data[key]['token_output'])
            tail_truncate = 0
            input_token_length = data[key]['input_length']
            if token_length > max_profile_token_length:
                tail_truncate = token_length - max_profile_token_length
                input_token_length = max(input_token_length - tail_truncate, 0)
                
            tokenized_info = {"input_ids": torch.tensor(data[key]['token_output']).reshape(1, -1)[:, tail_truncate:], 
                            "attention_mask": torch.tensor([1 for i in range(len(data[key]['token_output']))]).reshape(1, -1)
            }
            #pred_profiler.append(copy.deepcopy(tokenized_info))
            with torch.inference_mode():
                # If transformer.__version__ == v4.45.1, directly import it from transformers.cache_utils 
                past_key_values = extract_util.OffloadedCache() # GPU Memory Efficient
                tokenized_info["past_key_values"] = past_key_values
                try:
                    #snapshot = model.forward(**tokenized_info, output_attentions=True)
                    #snapshot, hook_info = recorder.forward(tokenized_info)
                    rank_per_line, y, _ = detection_model.predict_using_recorder(recorder, tokenized_info, input_token_length, candidate_tokens)
                except torch.cuda.OutOfMemoryError as e:
                    oom_keys.append(key)
                    torch.cuda.empty_cache()
                    logger.info(f"Out of Memory error occurred for key: {key}. Moving to next.")
                    continue
            before_act_dict[key] = copy.deepcopy(detection_model.cache.astype(np.float16))
            first_token_info_dict[key] = copy.deepcopy(detection_model.hidden_first.to(torch.float16))
        else:
            if extract_code:
                code_blocks_info = None
            else:
                code_blocks_info = [[0, len(data[key]["str_output"])]]
            candidate_tokens = dataset_utils.get_candidate_tokens(data, key, tokenizer, language, code_blocks_info=code_blocks_info)
            rank_per_line, y = detection_model.predict(x=before_act_dict[key], first_token=first_token_info_dict[key])
        #attn_snapshot = hook_info[layer]
        #del snapshot
        # Inference process of LBL baseline method
        #pred_result = detection_model.predict([attn_snapshot], input_token_length, candidate_tokens)
        #pred_profiler.append(pred_input)
        #pred_result = pred_result[:, 1]
        result[key] = [y, rank_per_line]
        #rank_per_line = sorted(list(zip(pred_result, [i for i in range(len(pred_result))])), reverse=True)[:topk]
        selected_token = set()
        for line in rank_per_line[:topk]:
            selected_token = selected_token.union(set(candidate_tokens[int(line)]))
        
        topk_score = compute_hit(important_token_info[key], selected_token)
        score += topk_score
        counter += 1
        
    #torch.save(pred_profiler, "pred_label_recoreder.pt")
    return score, counter, result, oom_keys, before_act_dict, first_token_info_dict
    
@app.command()
def main(
    important_label_path: Annotated[Path, typer.Option()],
    data_path: Annotated[Path, typer.Option()],
    config_file: Annotated[Path, typer.Option()],
    result_output_path: Annotated[Path, typer.Option()],
    parallel: Annotated[bool, typer.Option("--parallel/--no-parallel")] = True,
    detection_model_path: Annotated[Path, typer.Option()] = None,
    before_cache_save_folder: Annotated[Path, typer.Option()] = None,
):
    FALLBACK_ARGS = {
        "quantization": "4bit",
        "user_args": {"attn_implementation":"eager"},
        "device_map": "balanced"
    }
    # ===== Load configuration =====
    with open(config_file, 'r') as file:
        config_dict = yaml.safe_load(file)
    model_name = config_dict["llm_config"]["model_name"]
    quantization = config_dict["llm_config"]["quantization"]
    layer = config_dict["task_config"]["layer"]
    language = config_dict["task_config"]["language"]
    if detection_model_path is None:
        # Fallback policy
        detection_model_path = config_dict["task_config"]["detection_model_path"]
    mode = config_dict["task_config"]["mode"]
    max_profile_token_length = config_dict["task_config"]["max_profile_token_length"]
    extract_code = config_dict["task_config"]["extract_code"]
    collect_type = config_dict["task_config"]['collect_type']
    assert collect_type in ["attention", "hidden"]
    assert mode in ["lbl", "sae", "uncertainty"]
    
    cache_dir = None
    if "HF_HOME" in config_dict["system_setting"]:
        os.environ["HF_HOME"] = config_dict["system_setting"]["HF_HOME"]
    if "cache_dir" in config_dict["system_setting"]:
        cache_dir = config_dict["system_setting"]["cache_dir"]
    else:
        cache_dir = None
    FALLBACK_ARGS["model_name"] = model_name
    FALLBACK_ARGS["parallel"] = parallel
    FALLBACK_ARGS["cache_dir"] = cache_dir

    # ===== Load Data =====
    data = utils.load_shelve(data_path)
    tokenizer = utils.load_tokenizer(model_name)
    recorder = None

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
        if key not in error_line_info:
            continue
        if data[key]["code_correctness"] == 'correct':
            continue
        if len(error_line_info[key][0]) == 0:
            logger.info("No error info for key: {}".format(key))
            continue
        evaluate_key_list.append(key)

    if mode != "uncertainty":
        if before_cache_save_folder is not None:
            dataset_identifier = os.path.basename(data_path).split(".")[0]
            model_identifier = model_name.split("/")[-1]
            before_cache_save_path = os.path.join(before_cache_save_folder, f"{model_identifier}_{mode}_{collect_type}_extract_code{extract_code}_{dataset_identifier}_before_cache.pt")
            if os.path.exists(before_cache_save_path):
                before_info = torch.load(before_cache_save_path)
                before_dict = before_info["before_dict"]
                first_token_info_dict = before_info["first_token_info_dict"]
            else:
                before_dict = None
                first_token_info_dict = None
                # ===== Load Model =====
                model, tokenizer = utils.load_opensource_model(model_name, parallel=parallel, quantization=quantization, cache_dir=cache_dir)
                if collect_type == "attention":
                    recorder = extract_util.TransHookRecorder({layer: {"output_attentions": True}}, model)
                elif collect_type == "hidden":
                    recorder = extract_util.TransHookRecorder({layer: {"return_first": True}}, model, mode="plain")
        else:
            before_cache_save_path = None
            before_dict = None
            first_token_info_dict = None

            # ===== Load Model =====
            model, tokenizer = utils.load_opensource_model(model_name, parallel=parallel, quantization=quantization, cache_dir=cache_dir)
            if collect_type == "attention":
                recorder = extract_util.TransHookRecorder({layer: {"output_attentions": True}}, model)
            elif collect_type == "hidden":
                recorder = extract_util.TransHookRecorder({layer: {"return_first": True}}, model, mode="plain") 
        
        #if task.lower() == "defects4j":
        #    data_path = config_dict["task_config"]["repair_data_path"]
        #    loc_folder = config_dict["task_config"]["repair_loc_folder"]
        #    defects4j_path = config_dict["system_setting"]['DEFECTS4J_PATH']
        #    java_path = config_dict["system_setting"]['JAVA_PATH']
        #    dataset = defects4j.Defects4jDataset(data_path, loc_folder, defects4j_path, java_home=java_path)

        # ==== Begin collecting attention scores ====
        if mode == "lbl":
            detection_model = detect_model.LBLRegression.load(detection_model_path)
        elif mode == "sae":
            detection_model = detect_model.EncoderClassifier.load(detection_model_path)
            
        #lbl_model.load(lbl_model_path)
        
        score, counter, first_result, oom_keys, before_dict_1, first_token_info_dict_1 = \
        evaluate_binding(data, 
                        evaluate_key_list, 
                        important_token_info, 
                        recorder, 
                        tokenizer,
                        detection_model, 
                        language, 
                        layer,
                        max_profile_token_length=max_profile_token_length,
                        extract_code=extract_code,
                        before_act_dict=before_dict,
                        first_token_info_dict=first_token_info_dict
                        )
        
        second_result = dict()
        before_dict_2 = dict()
        first_token_info_dict_2 = dict()
        if len(oom_keys) > 0:
            logger.info("Begin fallback inference for OOM data points")
            del recorder
            del model
            time.sleep(3)
            gc.collect()
            torch.cuda.empty_cache()
            model, tokenizer = utils.load_opensource_model(**FALLBACK_ARGS)
            if collect_type == "attention":
                recorder = extract_util.TransHookRecorder({layer: {"output_attentions": True}}, model)
            elif collect_type == "hidden":
                recorder = extract_util.TransHookRecorder({layer: {"return_first": True}}, model, mode="plain") 
            score, counter, second_result, oom_keys, before_dict_2, first_token_info_dict_2 = \
                evaluate_binding(data, 
                                oom_keys, 
                                important_token_info, 
                                recorder, 
                                tokenizer,
                                detection_model, 
                                language, 
                                layer,
                                score=score,
                                counter=counter,
                                max_profile_token_length=max_profile_token_length,
                                extract_code=extract_code,
                                before_act_dict=before_dict,
                                first_token_info_dict=first_token_info_dict
                            )
            if len(oom_keys) > 0:
                logger.error("OOM error still exists after fallback inference. Ignore them.")
                logger.error("OOM keys: {}".format(oom_keys))
        
        #topk_recorder = {"top5": score / counter}
        result = {**first_result, **second_result}
        if before_dict is None and before_cache_save_path is not None:
            before_dict = {**before_dict_1, **before_dict_2}
            first_token_info_dict = {**first_token_info_dict_1, **first_token_info_dict_2}
            torch.save({"before_dict": before_dict, "first_token_info_dict": first_token_info_dict}, before_cache_save_path)
        else:
            del before_dict_1
            del before_dict_2
    else:
        result = dict()
        oom_keys = dict()
        
    topk_recorder = {"hit_rate": {}, "recall": {}, "hit_line_rate": {}}
    hit_detail = dict()
    for topk_iter in [1, 3, 5, 10]:
        hit_detail["top{}".format(topk_iter)] = dict()
        hit_score = 0
        hit_line_score = 0
        recall_score = 0
        counter = 0
        for key in evaluate_key_list:
            if extract_code:
                code_blocks_info = None
            else:
                code_blocks_info = [[0, len(data[key]["str_output"])]]
            candidate_tokens = dataset_utils.get_candidate_tokens(data, key, tokenizer, language, code_blocks_info=code_blocks_info)
            if mode != "uncertainty":
                recorded_result = result[key]
                rank_per_line = recorded_result[1][:topk_iter]
                selected_token = set()
                for line in rank_per_line:
                    selected_token = selected_token.union(set(candidate_tokens[line]))
            else:
                selected_token = utils.obtain_topk_tokens_by_prob(data, candidate_tokens, key, topk_iter, np.mean)
                result[key] = [None, list(selected_token)]
            
            cur_hit_score = compute_hit(important_token_info[key], selected_token)
            cur_hit_line_score = compute_hit_line(important_token_info[key], selected_token)
            cur_recall_score = compute_recall(important_token_info[key], selected_token)
            
            hit_detail["top{}".format(topk_iter)][key] = {"hit": cur_hit_score, 
                                                          "hit_line": cur_hit_line_score, 
                                                          "recall": cur_recall_score,
                                                          "gt": important_token_info[key],
                                                          "selected_token": list(selected_token)
                                                          }
            
            hit_score += cur_hit_score
            hit_line_score += cur_hit_line_score
            recall_score += cur_recall_score
            
            counter += 1
        topk_recorder['hit_rate']["top{}".format(topk_iter)] = hit_score / counter
        topk_recorder['recall']["top{}".format(topk_iter)] = recall_score / counter
        topk_recorder['hit_line_rate']["top{}".format(topk_iter)] = hit_line_score / counter
    
    if mode != "uncertainty":
        for key in result:
            if isinstance(result[key][0], torch.Tensor) or isinstance(result[key][0], np.ndarray):
                result[key][0] = result[key][0].tolist()
            if isinstance(result[key][1], torch.Tensor) or isinstance(result[key][1], np.ndarray):
                result[key][1] = result[key][1].tolist()
        
    
    with open(result_output_path, "w") as ofile:
        json.dump({"result": topk_recorder, "pred_result": result, "OOM_keys": oom_keys, "hit_detail": hit_detail}, ofile)

if __name__ == "__main__":
    app()
