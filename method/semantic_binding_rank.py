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
import method.semantic_binding as sb

app = typer.Typer(pretty_exceptions_show_locals=False, pretty_exceptions_short=False)
#app = typer.Typer(pretty_exceptions_short=False)

from utility.utils import HARD_TOKEN_LIMIT

    
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
    training_data_path_name = config_dict["task_config"]["training_data_path_name"]
    if encoder_path is None:
        encoder_path = config_dict['task_config']["encoder_path"]
    collect_type = config_dict["task_config"]['collect_type']
    dataset_list = config_dict["task_config"].get("dataset_list", ["humaneval"])
    data_source = config_dict["task_config"].get("data_source", ["shelve"])
    extract_code_list = config_dict["task_config"]["extract_code_list"]
    important_label_path_list = config_dict["task_config"].get("important_label_path_list", [None])
    data_path_list = config_dict["task_config"]['data_path_list']
    
    net_layer = config_dict["task_config"]['fit_model_param'].get("net_layer", 4)
    input_dim = config_dict["task_config"]['fit_model_param'].get("input_dim", 128)
    hidden_dim = config_dict["task_config"]['fit_model_param'].get("hidden_dim", 32)
    if agg is None:
        agg = config_dict["task_config"]['fit_model_param'].get("agg", None)
    if agg is None:
        enable_classifier = False
    else:
        enable_classifier = True
        logger.info(f"Enable Classifier Mode with {agg} aggregation")
    lr = config_dict["task_config"]['fit_model_param'].get("lr", 1e-4)
    lr = float(lr)
    batch_size = config_dict["task_config"]['fit_model_param'].get("batch_size", 2)
    batch_size = int(batch_size)
    num_epochs = config_dict["task_config"]['fit_model_param'].get("num_epochs", 20)
    num_epochs = int(num_epochs)
    weight_decay = config_dict["task_config"]['fit_model_param'].get("weight_decay", 1e-5)
    weight_decay = float(weight_decay)
    act = config_dict["task_config"]['fit_model_param'].get("act", "ReLU")
    beta1 = config_dict["task_config"]['fit_model_param'].get("beta1", 0.9)
    beta1 = float(beta1)
    
    logger.info(f"Training Rank model")
    #assert data_source in ["shelve", "tracer"]
    assert collect_type in ["hidden"]
    #assert dataset in ["humaneval", "safim"]
    FALLBACK_ARGS["model_name"] = model_name
    FALLBACK_ARGS["parallel"] = parallel
    FALLBACK_ARGS["cache_dir"] = cache_dir
    
    assert encoder_path is not None
    encoder_param = torch.load(encoder_path)
    encoder = sae_model.Autoencoder.from_state_dict(encoder_param["model_state_dict"])

    tokenizer = utils.load_tokenizer(model_name)
    data_line_token_pair = sb.load_gt(data_source, dataset_list, data_path_list, important_label_path_list, tokenizer)
    ## Load model
    quantization = config_dict["llm_config"]["quantization"]
    #generate_config = config_dict["llm_config"]["generate_config"]
    
    training_data_path = os.path.join(training_data_folder, training_data_path_name.format(layer))
    os.makedirs(training_data_path, exist_ok=True)
    data_collection_flag = False

    train_x = {key: {} for key in dataset_list}
    train_y = {key: {} for key in dataset_list}
    snapshot_x = {}
    store_y = {}
    first_token_dict = {}
    first_token_in_dict = {}
    candidate_token_dict = {}
    for data_class_idx, data_class_name in enumerate(dataset_list):
        cur_data_path = os.path.join(training_data_path, data_class_name)
        if os.path.exists(cur_data_path):
            logger.info(f"Load {collect_type} training data from {cur_data_path}")
            cur_train_x, cur_train_y, cur_first_token_dict, cur_first_token_in_dict, cur_candidate_token_dict = sb.load_from_existing_data(cur_data_path, 
                                                                                                                                        "sae",
                                                                                                                                        encoder
                                                                                                                                        )
            train_x[data_class_name] = cur_train_x
            train_y[data_class_name] = cur_train_y
            first_token_dict[data_class_name] = cur_first_token_dict
            candidate_token_dict[data_class_name] = cur_candidate_token_dict
            first_token_in_dict[data_class_name] = cur_first_token_in_dict
        else:
            if data_collection_flag is False:
                model, tokenizer = utils.load_opensource_model(model_name, parallel=parallel, quantization=quantization, cache_dir=cache_dir)
                data_collection_flag = True
            data = data_line_token_pair[0][data_class_idx]
            error_line_info = data_line_token_pair[1][data_class_idx]
            important_token_info = data_line_token_pair[2][data_class_idx]
            
            snapshot_x[data_class_name] = dict()
            candidate_token_dict[data_class_name] = dict()
            store_y[data_class_name] = dict()
            first_token_dict[data_class_name] = dict()
            first_token_in_dict[data_class_name] = dict()
            snap_shot = sb.get_hidden_snapshot(model, layer, data)
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
                _, before_latent_activations = collect_hidden_states(snap_shot_single, input_token_length, candidate_tokens, None)
                first_token_dict[data_class_name][key] = snap_shot_single[input_token_length]
                first_token_in_dict[data_class_name][key] = (candidate_tokens[0] == 0)
                with torch.inference_mode():
                    latent_activations, info = encoder.encode(before_latent_activations.to(encoder.device))
                snapshot_x[data_class_name][key] = before_latent_activations.cpu().numpy()
                train_x[data_class_name][key] = latent_activations.cpu().numpy()
                train_y[data_class_name][key] = label_dict[key]
                store_y[data_class_name][key] = label_dict[key]
            torch.save(
                {"snapshot_x": snapshot_x[data_class_name], 
                "store_y": store_y[data_class_name], 
                "first_token": [first_token_dict[data_class_name], first_token_in_dict[data_class_name]],
                "candidate_token_dict": candidate_token_dict[data_class_name],
                }, cur_data_path)
    

    train_x = flat_data_dict(train_x)
    train_y = flat_data_dict(train_y)
    first_token_dict = flat_data_dict(first_token_dict)
    candidate_token_dict = flat_data_dict(candidate_token_dict)

    
    if agg is None:
        # Ranking loss is not well-defined
        # for correct code snippet, we remove them from the training data
        iter_list = list(train_y.keys())
        for key in iter_list:
            if sum(train_y[key]) == 0:
                train_x.pop(key)
                train_y.pop(key)
                first_token_dict.pop(key)
                candidate_token_dict.pop(key)
    
    logger.info(f"Train X Length: {len(train_x)}")
    

    clf = detect_model.RankNet(
        input_dim,
        net_layer,
        hidden_dim,
        enable_classifier = enable_classifier,
        act = act
    )
    
    learning_param = {
        "lr": lr,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "weight_decay": weight_decay,
        "beta1": beta1,
        
    }
    
    if enable_classifier:
        learning_param["freeze"] = False
        learning_param["agg"] = agg
        clf.fit_classifier(
            train_x,
            train_y,
            candidate_token_dict,
            learning_param=learning_param,
            verbose=True
        )
    else:
        learning_param["loss_fn"] = "neuralNDCG"
        clf.fit(
            train_x,
            train_y,
            candidate_token_dict,
            learning_param=learning_param,
            verbose=True
        )

    classifier_param = {
        "param": None,
        "model_type": "ranker",
        "dim_red": None,
        "device": "cpu",
        "encoder_type": "Autoencoder",
        "encoder": encoder.state_dict(),
        "model": clf.pack()
        
    }
    torch.save(classifier_param, model_save_path)
    logger.info(f"Training finished. Model saved at {model_save_path}")
    
    
if __name__ == "__main__":
    app()
    