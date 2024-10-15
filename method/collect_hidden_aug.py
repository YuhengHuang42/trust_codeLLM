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
import copy
import random
import math

import sys
project_root = Path(__file__).resolve().parent.parent
# Add the project root to sys.path
sys.path.append(str(project_root))

import utility.utils as utils
from method.extract import extract_util
from method.extract import naive_store
from method.collect_hidden import PretrainCodedata


def remove_empty_lines(input_string):
    # Split the string into lines
    lines = input_string.splitlines()
    # Filter out empty lines
    non_empty_lines = [line for line in lines if line.strip() != "" and line.strip().startswith("```") == False]
    # Join the non-empty lines back into a single string
    return "\n".join(non_empty_lines)

class AugPretrainCodedata(PretrainCodedata):
    def __init__(self,
                 mutate_prop: float, 
                 dataset_id: str,
                 feature_key_list: list=["java", "python", "c++", "javascript"],
                 preprocess_all_in_memory=False,
                 split_token="\n",
                 original_label=1,
                 mutated_label=2
                 ):
        """
        
        ---
        Args:
            mutate_prop: float. The proportion of data to be mutated.
            preprocess_all_in_memory: bool. If True, preprocess all data in memory before training.
                This flag will result in a totally different behaviors in this class 
                (e.g., data length, returned item). 
        """
        # datasets.arrow_dataset.Dataset
        super().__init__(dataset_id, feature_key_list, preprocess_all_in_memory, split_token)
        self.mutate_prop = mutate_prop
        self.original_label = original_label
        self.mutated_label = mutated_label
        if preprocess_all_in_memory:
            self.aug_data_all()
        
    def _get_mutated_code(self, idx, code, code_split_pos, mutated_pos_list, mutated_op_list):
        """
        ---
        Args:
            idx: int. The index of the data.
            code: str. The original code.
            code_split_pos: List[int]. The split position ("\n") of the original code.
            mutated_pos_list: List[int]. The index of the original pos_list to be mutated.
            mutated_op_list: List[str]. The operator to be used for mutation.
        Return:
            new_mutated: str. The mutated code.
            new_mutated_code_split_pos: List[int]. The split position ("\n") of the mutated code.
            labels: List[int]. The label of each code line.
            contrastive_pair: List[List]. The contrastive pair of the original and mutated code. Could contain repeated index.
                The item inside the list refers to the position relative to the code_split_pos.
        """
        new_mutated = ""
        new_mutated_code_split_pos = []
        labels = []
        contrastive_original = [] # Original, Mutated
        contrastive_mutated = []
        mutated_pos_index_list = []
        mutated_set = set(mutated_pos_list)
        mutated_pointer = 0

        last_split_pos = -1
        #NON_DROP = 1
        #DROP = 2
        #mutated_recorder = []
        drop_label_idx_recorder = []
        for idx, now_split_pos in enumerate(code_split_pos):
            if idx in mutated_set:
                # Mutate the code
                op = mutated_op_list[mutated_pointer]
                if op == "switch_inside":
                    cur_code = self.switch_inside(code, code_split_pos, idx, self.select_line_according_to_split(code, last_split_pos, now_split_pos))
                elif op == "switch_outside":
                    cur_code = self.switch_outside(idx, self.select_line_according_to_split(code, last_split_pos, now_split_pos))
                elif op == "delete_line":
                    cur_code = self.delete_line()
                new_mutated += cur_code
                if len(cur_code) > 0:
                    # If it is not drop operator
                    new_mutated_code_split_pos.append(len(new_mutated)-1)
                    labels.append(self.mutated_label)
                    #mutated_recorder.append(NON_DROP)
                else:
                    # If it is drop operator
                    # labels[-1] = self.mutated_label
                    #mutated_recorder.append(DROP)
                    drop_label_idx_recorder.append(len(new_mutated_code_split_pos))
                #contrastive_original.append(now_split_pos)
                contrastive_original.append(idx)
                mutated_pos_index_list.append(len(new_mutated_code_split_pos)-1)
                last_split_pos = code_split_pos[idx]
                mutated_pointer += 1
            else:
                # Copy-paste the original code line.
                # split_pos records the exact position of "\n"
                # To include the actual code line, we need to add the length of split_token ("\n")
                #cur_code = code[last_split_pos+len(self.split_token): now_split_pos+len(self.split_token)]
                cur_code = self.select_line_according_to_split(code, last_split_pos, now_split_pos)
                new_mutated += cur_code
                new_mutated_code_split_pos.append(len(new_mutated)-1)
                last_split_pos = code_split_pos[idx]
                labels.append(self.original_label)
        #for idx, indicator in enumerate(mutated_recorder):
        #    if indicator == NON_DROP:
        #        contrastive_mutated.append(new_mutated_code_split_pos[idx])
        #    elif indicator == DROP:
        #        mutate_pos = min(mutated_pos_index_list[idx] + 1, len(new_mutated_code_split_pos)-1)
        #        contrastive_mutated.append(new_mutated_code_split_pos[mutate_pos])
        try:
            assert len(labels) > 0
        except:
            logger.error("Code Mutation Error: No Code Left")
            logger.error("labels", labels)
            logger.error("mutated_op_list", mutated_op_list)
            logger.error("new_mutated_code_split_pos", new_mutated_code_split_pos)
            logger.error("new_mutated", new_mutated)
            logger.error("original code", code)
            raise Exception
        drop_repeat_map = dict()
        for drop_idx in drop_label_idx_recorder:
            drop_idx = min(drop_idx, len(labels)-1)
            if drop_idx not in drop_repeat_map:
                drop_repeat_map[drop_idx] = 1
            if labels[drop_idx] == self.mutated_label:
                drop_repeat_map[drop_idx] += 1
            labels[drop_idx] = self.mutated_label
        for idx, label in enumerate(labels):
            if label == self.mutated_label:
                contrastive_mutated += [idx] * drop_repeat_map.get(idx, 1)
        assert len(contrastive_original) == len(contrastive_mutated)
        return new_mutated, new_mutated_code_split_pos, labels, [contrastive_original, contrastive_mutated]

    def aug_data_single(self, index):
        if self.preprocess_all_in_memory:
            item = self.processed_data[index]
            item_list = [item]
        else:
            item = self.data[index]
            item_list = self._preprocess_data_single(item) # Return List
        for item in item_list:
            code = item['output']['code']
            code_split_pos = item['output']['code_split_pos']
            mutated_pos_list = [i for i in range(len(code_split_pos))]
            random.shuffle(mutated_pos_list)
            mutated_num = max(math.ceil(len(mutated_pos_list) * self.mutate_prop), 1)
            mutated_pos_list = mutated_pos_list[:mutated_num]
            if mutated_num <= 2:
                # To avoid no code left error
                mutated_op_list = ["switch_outside"] * mutated_num
            else:
                mutated_op_list = random.choices(["switch_inside", "switch_outside", "delete_line"], k=mutated_num)
                if set(mutated_op_list) == {"delete_line"}:
                    mutated_op_list[0] = random.choices(["switch_inside", "switch_outside"], k=1)[0]
            new_mutated, new_mutated_code_split_pos, labels, contrastive_pair = self._get_mutated_code(index, code, code_split_pos, mutated_pos_list, mutated_op_list)
            item["mutated_code"] = {
                "code": new_mutated,
                "code_split_pos": new_mutated_code_split_pos,
                "line_label": labels,
                "contrastive_pair": contrastive_pair,
                "mutated_op_list": mutated_op_list
            }
        return item_list
    
    def aug_data_all(self):
        # TODO: Introduce multi-processing
        if not self.preprocess_all_in_memory:
            self._preprocess_data()
            self.preprocess_all_in_memory = True
        for idx in tqdm.tqdm(range(len(self.processed_data))):
            self.processed_data[idx] = self.aug_data_single(idx)[0]
            
    
    def select_line_according_to_split(self, code, previous_pos, cur_pos):
        real_start_pos = previous_pos + len(self.split_token) if previous_pos >= 0 else 0
        return code[real_start_pos: cur_pos+len(self.split_token)]
    
    def switch_inside(self, cur_code, code_split_pos, line_idx, cur_line):
        code_split_pos = copy.deepcopy(code_split_pos)
        code_split_pos.pop(line_idx)
        random_select_list = [num for num in range(0, len(code_split_pos))]
        random.shuffle(random_select_list)
        for item in random_select_list:
            end_pos = code_split_pos[item]
            if item - 1 == 0:
                start_pos = -1
            else:
                start_pos = code_split_pos[item-1]
            select_line = self.select_line_according_to_split(cur_code, start_pos, end_pos)
            if select_line.strip() != cur_line.strip() and select_line.strip() != "":
                return select_line
        return ""
    
    def switch_outside(self, data_idx, cur_line):
        if self.preprocess_all_in_memory:
            valid_numbers = [num for num in range(0, len(self.processed_data)) if num != data_idx]
        else:
            valid_numbers = [num for num in range(0, len(self.data)) if num != data_idx]
        random.shuffle(valid_numbers)
        for chosen_idx in valid_numbers:
            if self.preprocess_all_in_memory:
                processed_data = self.processed_data[chosen_idx]
            else:
                processed_data_list = self._preprocess_data_single(self.data[chosen_idx]) # Return a list
                processed_data = random.choices(processed_data_list, k=1)[0]
            other_code_split_pos = copy.deepcopy(processed_data['output']['code_split_pos'])
            cur_code = processed_data['output']['code']
            random_select_list = [num for num in range(0, len(other_code_split_pos))]
            random.shuffle(random_select_list)
            for item in random_select_list:
                end_pos = other_code_split_pos[item]
                if item - 1 < 0:
                    start_pos = -1
                else:
                    start_pos = other_code_split_pos[item-1]
                select_line = self.select_line_according_to_split(cur_code, start_pos, end_pos)
                if select_line.strip() != cur_line.strip() and select_line.strip() != "":
                    return select_line
        return ""
        
    
    def delete_line(self):
        return ""
    
    def __getitem__(self, index):
        if self.preprocess_all_in_memory:
            item = self.processed_data[index]
        else:
            item = self.data[index]
            item = self.aug_data_single(item)
            
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
                token_num += len(item['mutated_code']['code_split_pos'])
        else:
            for idx, item in enumerate(self.data):
                item_list = self.aug_data_single(idx)
                for item in item_list:
                    token_num += len(item['output']['code_split_pos'])
                    token_num += len(item['mutated_code']['code_split_pos'])
        return token_num

def wrap_input(item, code, code_split_pos, tokenizer, split_token, contrastive_list):
    def adjust_indices(sub_indicator_list, removal_indices):
        # Create a set for fast lookup of removal indices
        removal_indices_set = set(removal_indices)
        
        # Adjust sub_indicator_list by accounting for removed indices
        adjusted_sub_indicator_list = []
        for index in sub_indicator_list:
            # If the index was removed, skip it
            if index in removal_indices_set:
                continue
            
            # Count how many previous elements have been removed before the current index
            shift = sum(1 for ri in removal_indices if ri < index)
            
            # Adjust the index by subtracting the number of removed elements before it
            adjusted_sub_indicator_list.append(index - shift)
        
        return adjusted_sub_indicator_list
    
    context = item['context']
    code_description = item['output']['code_description']
    problem_description = item['output']['problem_description']
    description = problem_description + code_description
    target_input = context + description + "\n```\n" + code
    context_str_len = len(context + description + "\n```\n")
    input_info = tokenizer(target_input, return_tensors="pt", return_offsets_mapping=True)
    offset_mapping = input_info["offset_mapping"].squeeze().tolist()
    input_info.pop("offset_mapping")
    split_tok_pos = list()
    removal_indices = list()
    for idx, pos in enumerate(code_split_pos):
        real_pos = pos + context_str_len
        try:
            split_tok_pos.append(utils.match_token_in_offset_mapping(offset_mapping, real_pos, real_pos+len(split_token))[0])
        except:
            logger.warning(f"Cannot find split token for {pos} in {item['id']}")
            removal_indices.append(idx)
            continue
    if len(split_tok_pos) == 0:
        split_tok_pos = [-1]
    
    contrastive_list = adjust_indices(contrastive_list, removal_indices)
    return input_info, split_tok_pos, contrastive_list
    
def inference_and_collect(
    dataset,
    recorder,
    tokenizer,
    store,
    run_original: bool,
):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x:x)
    for idx, item in enumerate(tqdm.tqdm(data_loader)):
        item = item[0]
        input_info, original_split_tok_pos, contrastive_list_original = wrap_input(item, 
                                                                        item['output']['code'],
                                                                        item['output']['code_split_pos'], 
                                                                        tokenizer, 
                                                                        dataset.split_token, 
                                                                        item['mutated_code']['contrastive_pair'][0]
                                                                )
        if run_original:
            _, record_dict = recorder.forward(input_info)
            
            hidden_states = record_dict[list(recorder.layer_names.keys())[0]] # Only one layer
            hidden_states = hidden_states[0] # (1, token_num, hidden_size)
            hidden_states = hidden_states[0, original_split_tok_pos, :] # (split_token_num, hidden_size)
            recorder.clear_cache()
            
            store_info = {
                "original": {"hidden_states": hidden_states},
            }
        else:
            store_info = store[idx]
        if "mutated_code" not in store_info:
            store_info["mutated_code"] = {}
        
        input_info, mutated_split_tok_pos, contrastive_list_mutated = wrap_input(item,
                                                                                 item['mutated_code']['code'],
                                                                                 item['mutated_code']['code_split_pos'],
                                                                                 tokenizer,
                                                                                 dataset.split_token,
                                                                                 item['mutated_code']['contrastive_pair'][1]
                                                                                 )
        _, record_dict = recorder.forward(input_info)
        
        hidden_states = record_dict[list(recorder.layer_names.keys())[0]] # Only one layer
        hidden_states = hidden_states[0] # (1, token_num, hidden_size)
        hidden_states = hidden_states[0, mutated_split_tok_pos, :] # (split_token_num, hidden_size)
        
        key_list = [int(i) for i in list(store_info["mutated_code"].keys())]
        if len(key_list) == 0:
            cur_key = str(0)
        else:
            cur_key = max(key_list) + 1
            cur_key = str(cur_key)
        store_info["mutated_code"][cur_key] = {"hidden_states": hidden_states, 
                                               "contrastive_list_original": contrastive_list_original,
                                               "contrastive_list_mutated": contrastive_list_mutated, 
                                               "line_label": item['mutated_code']['line_label']}
        if run_original:
            store.append(store_info)
        else:
            store.set_item(idx, store_info)
    
    store.save_to_disk()
            
            
        
# python3 method/collect_hidden_aug.py --config-file model_eval_config/CodeLlama_hidden_state.yaml --result-output-path /data/data_disk/trust_code/leetcode_aug3_fix
app = typer.Typer(pretty_exceptions_show_locals=False, pretty_exceptions_short=False)
@app.command()
def main(
    config_file: Annotated[Path, typer.Option()],
    result_output_path: Annotated[Path, typer.Option()],
    parallel: Annotated[bool, typer.Option("--parallel/--no-parallel")] = True,
    #aug_times: Annotated[int, typer.Option("--aug-times")] = 1
):
    start = time.time()
    # ===== Load configuration =====
    with open(config_file, 'r') as file:
        config_dict = yaml.safe_load(file)
    model_name = config_dict["llm_config"]["model_name"]
    quantization = config_dict["llm_config"]["quantization"]
    hidden_layer = config_dict["task_config"]["hidden_layer"]
    split_token = config_dict['task_config']["split_token"]
    dataset_name = config_dict['task_config']["dataset_name"]
    mutation_prop = (config_dict['task_config']["mutation_prop"])
    feature_key_list = config_dict['task_config']["feature_key_list"]
    
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
    
    aug_times = len(mutation_prop)
    data = AugPretrainCodedata(
        float(mutation_prop[0]),
        dataset_name,
        feature_key_list,
        preprocess_all_in_memory=True,
        split_token=split_token
    )
    #instance_real_num = data.compute_all_inference_num()
    #store = naive_store.NaiveTensorStore()
    store = naive_store.VariedKeyTensorStore()
    store.init(
        save_dir = result_output_path
    )
    
    inference_and_collect(
        data,
        recorder,
        tokenizer,
        store,
        True
    )
    
    for i in range(1, aug_times):
        # dataset.aug_data_all()
        data = AugPretrainCodedata(
        float(mutation_prop[i]),
        dataset_name,
        feature_key_list,
        preprocess_all_in_memory=True,
        split_token=split_token
    )
        inference_and_collect(
            data,
            recorder,
            tokenizer,
            store,
            False
        )
    
    end = time.time()
    logger.info(f"Total time: {end - start}")

if __name__ == "__main__":
    app()
    