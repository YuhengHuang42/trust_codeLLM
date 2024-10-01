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
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn import svm

import sys
project_root = Path(__file__).resolve().parent.parent
# Add the project root to sys.path
sys.path.append(str(project_root))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
import utility.utils as utils
from task import dataset_utils

app = typer.Typer(pretty_exceptions_show_locals=False, pretty_exceptions_short=False)

def collect_attn_snapshot(model, tokenizer, data, cleaner=None):
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

class LBLRegression():
    def __init__(self):
        self.fit_model_param = None
        self.clf = None
        self.attn_layer = None
        self.model_type = None
        
    def fit(self, train_x, train_y, model_type, fit_model_param, attn_layer):
        self.fit_model_param = fit_model_param
        self.attn_layer = attn_layer
        self.model_type = model_type
        if self.model_type.lower() == "logistic":
            self.clf = LogisticRegression(**fit_model_param).fit(train_x, train_y)
        elif self.model_type.lower() == "svm":
            self.clf = svm.SVC(probability=True, **fit_model_param).fit(train_x, train_y)
        
    def save(self, path):
        joblib.dump({"model": self.clf, "param": self.fit_model_param, "attn_layer": self.attn_layer, "model_type": self.model_type}, path)
    
    def load(self, path):
        loaded_info = joblib.load(path)
        self.clf = loaded_info["model"]
        self.fit_model_param = loaded_info["param"]
        self.layer = loaded_info["attn_layer"]
        self.model_type = loaded_info["model_type"]
    
    def predict(self, attn_snapshot, input_token_length, candidate_tokens):
        al_result = utils.collect_attention_map(attn_snapshot, self.layer, input_token_length, candidate_tokens)
        x = np.stack(al_result)
        y = self.clf.predict_proba(x)
        return y

'''
regr = svm.SVR()
regr.fit(X, y)
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
    attn_layer = config_dict["task_config"]["attn_layer"]
    fit_model_param = config_dict["task_config"]["fit_model_param"]
    start_token = config_dict["llm_config"]["start_token"]
    
    ## Load model
    quantization = config_dict["llm_config"]["quantization"]
    #generate_config = config_dict["llm_config"]["generate_config"]
    tokenizer = utils.load_tokenizer(model_name)

    target_buggy_positions, important_token_info = utils.get_important_token_pos(error_line_info, data, tokenizer)
    
    training_data_path = os.path.join(training_data_folder, "layer_{}_attn_training_data.pt".format(attn_layer))
    if os.path.exists(training_data_path):
        logger.info(f"Load attention training data from {training_data_path}")
        data_collection_flag = True
    else:
        logger.info(f"Attention Training data not found at {training_data_path}")
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
        cleaner = utils.ModelOutputCleaner(start_token, model_name)
        attn_data = collect_attn_snapshot(model, tokenizer, data, cleaner)
        label_dict = dict()
        for key in data:
            candidate_tokens = dataset_utils.get_candidate_tokens(data, key, tokenizer, "python", code_blocks_info=[[0, len(data[key]["str_output"])]])
            if key in important_token_info:
                labels = utils.label_seg(important_token_info[key], candidate_tokens)
            else:
                labels = [0 for i in range(len(candidate_tokens))]
            label_dict[key] = labels
            attn_snapshot = attn_data[key]
            input_token_length = data[key]["input_length"]
            al_result = utils.collect_attention_map(attn_snapshot, attn_layer, input_token_length, candidate_tokens)
            train_x += [i for i in al_result]
            train_y += label_dict[key]
        torch.save({"train_x": train_x, "train_y": train_y}, training_data_path)
    
    #for key in data:
    #    input_token_length = data[key]["input_length"]
    #    attn_snapshot = attn_data[key]
    #    al_result = utils.collect_attention_map(attn_snapshot, attn_layer, input_token_length, candidate_tokens)
    #    train_x += [i for i in al_result]
    #    train_y += label_dict[key]
    train_x = np.stack(train_x)
    
    clf = LBLRegression("svm")
    clf.fit(train_x, train_y, fit_model_param, attn_layer)
    clf.save(model_save_path)
    
    
if __name__ == "__main__":
    app()
    