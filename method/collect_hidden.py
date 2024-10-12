import datasets
from torch.utils.data import Dataset
from loguru import logger
import typer
from typing_extensions import Annotated
from pathlib import Path
import yaml
import os
import torch
import tqdm
import time

#import sys
#project_root = Path(__file__).resolve().parent.parent
# Add the project root to sys.path
#sys.path.append(str(project_root))

import utility.utils as utils
from method.extract import extract_util
from method.extract import naive_store



class PretrainCodedata(Dataset):
    def __init__(self, 
                 data: datasets.arrow_dataset.Dataset,
                 feature_key_list: list=["java", "python", "c++", "javascript"],
                 preprocess_all_in_memory=False,
                 split_token="\n"
                 ):
        """
        
        ---
        Args:
            preprocess_all_in_memory: bool. If True, preprocess all data in memory before training.
                This flag will result in a totally different behaviors in this class 
                (e.g., data length, returned item). 
        """
        self.data = data
        self.processed_data = None
        self.feature_key_list = feature_key_list
        self.preprocess_all_in_memory = preprocess_all_in_memory
        self.split_token = split_token
        if self.preprocess_all_in_memory:
            self._preprocess_data()
        
    
    
    def _preprocess_data_single(self, item):
        return_dict = {"context": None, "output": {}}
        return_dict['context'] = item["content"]
        for key in self.feature_key_list:
            codes, code_pos_infos = utils.extract_code_block(item[key])
            if len(codes) < 1:
            #    logger.warning(f"Code block not found in {key} of {item['id']}")
            #    logger.warning("Skip this item")
                continue
            code = codes[-1]
            code_pos_info = code_pos_infos[-1]
            code_description_start_pos = utils.find_all_occurrences("\n", item[key][code_pos_info[-1]:])[0]
            code_description_start_pos = code_description_start_pos + code_pos_info[-1]
            code_description = item[key][code_description_start_pos:]
            split_pos = utils.find_all_occurrences(self.split_token, code)
            previous_pos = 0
            processed_split_pos = []
            for pos in split_pos:
                if code[previous_pos:pos].strip() == "":
                    previous_pos = pos
                    continue
                else:
                    processed_split_pos.append(pos)
                    previous_pos = pos
            return_dict["output"][key] = {"code": code, 
                                          "code_description": code_description, 
                                          "code_split_pos": processed_split_pos
                                          }
        return return_dict
    
    def _preprocess_data(self):
        self.processed_data = []
        for item in self.data:
            processed_item = self._preprocess_data_single(item)
            for key in processed_item["output"]:
                self.processed_data.append({"context": processed_item["context"], "output": processed_item["output"][key]})
    
    def __getitem__(self, index):
        if self.preprocess_all_in_memory:
            item = self.processed_data[index]
        else:
            item = self.data[index]
            item = self._preprocess_data(item)
        return item

    def __len__(self):
        if self.preprocess_all_in_memory:
            return len(self.processed_data)
        else:
            return len(self.data)
    
    def compute_all_inference_num(self):
        token_num = 0
        if self.preprocess_all_in_memory:
            for item in self.processed_data:
                token_num += len(item['output']['code_split_pos'])
        else:
            for item in self.data:
                item = self._preprocess_data(item)
                for key in self.feature_key_list:
                    token_num += len(item['output'][key]['code_split_pos'])
        return token_num

def inference_and_collect(
    dataset,
    recorder,
    tokenizer,
    store
):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x:x)
    for item in tqdm.tqdm(data_loader):
        item = item[0]
        context = item['context']
        code_description = item['output']['code_description']
        code = item['output']['code']
        target_input = context + code_description + "\n```\n" + code
        context_str_len = len(context + code_description + "\n```\n")
        input_info = tokenizer(target_input, return_tensors="pt", return_offsets_mapping=True)
        offset_mapping = input_info["offset_mapping"].squeeze().tolist()
        input_info.pop("offset_mapping")
        split_tok_pos = list()
        for pos in item['output']['code_split_pos']:
            real_pos = pos + context_str_len
            try:
                split_tok_pos.append(utils.match_token_in_offset_mapping(offset_mapping, real_pos, real_pos+len(dataset.split_token))[0])
            except:
                logger.warning(f"Cannot find split token for {pos} in {item['id']}")
                continue
        if len(split_tok_pos) == 0:
            split_tok_pos = [-1]
        _, record_dict = recorder.forward(input_info)
        

        hidden_states = record_dict[list(recorder.layer_names.keys())[0]] # Only one layer
        hidden_states = hidden_states[0] # (1, token_num, hidden_size)
        hidden_states = hidden_states[0, split_tok_pos, :] # (split_token_num, hidden_size)
        recorder.clear_cache()
        for i in range(hidden_states.shape[0]):
            store.append({"extracted_states": hidden_states[i], "labels": 1})
    
    store.save_to_disk()
            
            
        
# python3 collect_hidden.py --config-file model_eval_config/CodeLlama_hidden_state.yaml --result-output-path /data/huangyuheng/trust_code/codellama34b/hidden_state
app = typer.Typer(pretty_exceptions_show_locals=False, pretty_exceptions_short=False)
@app.command()
def main(
    config_file: Annotated[Path, typer.Option()],
    result_output_path: Annotated[Path, typer.Option()],
    parallel: Annotated[bool, typer.Option("--parallel/--no-parallel")] = True,
):
    start = time.time()
    # ===== Load configuration =====
    with open(config_file, 'r') as file:
        config_dict = yaml.safe_load(file)
    model_name = config_dict["llm_config"]["model_name"]
    quantization = config_dict["llm_config"]["quantization"]
    hidden_layer = config_dict["task_config"]["hidden_layer"]
    hidden_neuron = config_dict["task_config"]["hidden_neuron"]
    dataset_name = config_dict['task_config']["dataset_name"]
    split_token = config_dict['task_config']["split_token"]
    store_storage_multiplier = config_dict['task_config']["store_storage_multiplier"]
    
    cache_dir = None
    if "HF_HOME" in config_dict["system_setting"]:
        os.environ["HF_HOME"] = config_dict["system_setting"]["HF_HOME"]
    if "cache_dir" in config_dict["system_setting"]:
        cache_dir = config_dict["system_setting"]["cache_dir"]
    else:
        cache_dir = None
    # ===== Load Model =====
    model, tokenizer = utils.load_opensource_model(model_name, parallel=parallel, quantization=quantization, cache_dir=cache_dir)
    recorder = extract_util.TransHookRecorder({hidden_layer: {"return_first": True}}, model, mode="plain")
    
    leetcode_data = datasets.load_dataset(dataset_name)
    data = PretrainCodedata(
        leetcode_data["train"],
        preprocess_all_in_memory=True,
        split_token=split_token
    )
    instance_real_num = data.compute_all_inference_num()
    store = naive_store.NaiveTensorStore()
    store.init(
        allocated_size = round(instance_real_num * store_storage_multiplier), 
        config = {
                "extracted_states": {"shape": (hidden_neuron), "dtype": "float"},
                "labels": {"shape": (), "dtype": "int"}
            },
        save_dir = result_output_path
        )
    
    inference_and_collect(
        data,
        recorder,
        tokenizer,
        store
    )
    
    end = time.time()
    logger.info(f"Total time: {end - start}")

if __name__ == "__main__":
    app()
    