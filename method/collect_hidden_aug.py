from torch.utils.data import Dataset
import datasets
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
import pandas as pd 
import numpy as np
import subprocess
import shutil
import re
import json
from datasets import load_dataset
import difflib

import sys
project_root = Path(__file__).resolve().parent.parent
# Add the project root to sys.path
sys.path.append(str(project_root))

import utility.utils as utils
from method.extract import extract_util
from method.extract import naive_store
from method.collect_hidden import PretrainCodedata

class HumanEvalPack:
    def __init__(self,
                 dataset_name: str="bigcode/humanevalpack",
                 feature_key_list: list=["python", "js", "java", "go", "cpp", "rust"],
                 split_token="\n",
                 original_label=1,
                 mutated_label=2,
                ):
        self.feature_key_list = feature_key_list
        self.split_token = split_token
        self.data = []
        self.original_label = original_label
        self.mutated_label = mutated_label
        for key in self.feature_key_list:
            ds = load_dataset(dataset_name, key)["test"]
            self.data += ds.to_list()
    
        self.processed_data = []
        for item in self.data:
            self.processed_data.append(self.process_single(item, split_token="\n", original_label=original_label, mutated_label=mutated_label))
    
    def __getitem__(self, index):
        item = self.processed_data[index]
        return item

    def __len__(self):
        return len(self.processed_data)
    
    def find_contrastive_index(self, correct_lines, buggy_lines):
        """
        return [INDEX of correct lines, INDEX of buggy lines] of the diff results on correct_lines and buggy_lines
        This function has some hypotheses on the dataset characteristics (e.g., usually we get an equal tag after delete & insert), 
        so it needs to be very careful when modified for use on other datasets.
        """
        result = []
        # Initialize the SequenceMatcher
        matcher = difflib.SequenceMatcher(None, correct_lines, buggy_lines)

        # Get the opcodes which describe how to transform correct_lines into buggy_lines
        opcodes = matcher.get_opcodes()

        # Iterate over the opcodes and handle each case
        for op_idx, (tag, i1, i2, j1, j2) in enumerate(opcodes):
            if tag == 'equal':
                # Lines are the same in both files; no action needed for your requirement
                continue
            elif tag == 'replace':
                # Lines have been replaced
                #for i in range(i1, i2):
                #replace_correct_line = max(0, i2-1)
                #replace_buggy_line = max(0, j2-1)
                replace_correct_line = i1
                replace_buggy_line = j1
                result.append([replace_correct_line, replace_buggy_line, "replacce"])
                #print(f"Line {replace_correct_line}: correct (replaced) - {correct_lines[replace_correct_line]}")
                #print(f"Line {replace_buggy_line}: buggy (replacement) - {buggy_lines[replace_buggy_line]}")
            elif tag == 'delete':
                # Lines deleted from the correct code in the buggy code
                # (Correct code line at the end of the chunk that has been deleted, Error code line at the deleted position)
                match_flag = False
                if (op_idx + 1) < len(opcodes):
                    # Directly used the next matched one for contrastive learning
                    # Directly used the next matched one for contrastive
                    n_tag, i1, __, j1, ___ = opcodes[op_idx + 1]
                    if n_tag == "equal":
                        correct_line_num = i1 - 1
                        buggy_line_num = j1 - 1
                        match_flag = True
                        correct_line = correct_lines[correct_line_num].strip()
                        buggy_line = buggy_lines[buggy_line_num].strip()
                if (match_flag is False) and (i2 - i1 > 0):
                    # Correct code line at the end of the deleted chunk
                    correct_line_num = i2 - 1  # Indexing starts from 0

                    # Initialize correct_line
                    correct_line = correct_lines[correct_line_num].strip()

                    # If the correct line is empty, fall back to previous non-empty lines
                    #while len(correct_line) == 0 and correct_line_num > 0:
                    #    correct_line_num -= 1
                    #    correct_line = correct_lines[correct_line_num].strip()

                    # Buggy code line just before the deletion (context line)
                    #if j1 - 1 >= 0:
                    #    buggy_line_num = j1 - 1
                    #    buggy_line = buggy_lines[buggy_line_num].strip()
                    #else:
                    #    # If deletion is at the start, use the first line
                    #    buggy_line_num = 0
                    #    buggy_line = buggy_lines[buggy_line_num].strip()
                    if j1 - 1 >= 0 and len(buggy_lines[j1 - 1].strip()) == 0:
                        buggy_line_num = j1 - 1
                    else:
                        buggy_line_num = j1
                    buggy_line = buggy_lines[buggy_line_num].strip()

                result.append([correct_line_num, buggy_line_num, "delete"])
            elif tag == 'insert':
                # Lines inserted in the buggy code
                # (Correct code line at the position where the new lines are inserted, Error code line at the end of the inserted content)
                match_flag = False
                if (op_idx + 1) < len(opcodes):
                    # Directly used the next matched one for contrastive learning
                    n_tag, i1, __, j1, ___ = opcodes[op_idx + 1]
                    if n_tag == "equal":
                        correct_line_num = i1 - 1
                        buggy_line_num = j1 - 1
                        match_flag = True
                if (match_flag is False) and (j2 - j1 > 0):
                    # Correct code line at the insertion point
                    if i1 < len(correct_lines):
                        correct_line_num = i1  # Indexing starts from 0
                    else:
                        # If insertion is at the end, use the last line
                        correct_line_num = len(correct_lines) - 1
                        correct_line = correct_lines[correct_line_num].strip()
                    # Error code line at the end of the inserted content
                    buggy_line_num = j2 - 1  # Indexing starts from 0

                correct_line = correct_lines[correct_line_num].strip()
                buggy_line = buggy_lines[buggy_line_num].strip()
                result.append([correct_line_num, buggy_line_num, "insert"])
                #print(f"Insert - Correct line {correct_line_num}: '{correct_line}'")
                #print(f"Insert - Buggy line {buggy_line_num}: '{buggy_line}'\n")
        return result

    def process_single(self, dp, split_token="\n", original_label=1, mutated_label=2):
        problem_description = dp["prompt"]
        code_description = ""
        correct_code = remove_empty_lines(dp['canonical_solution'])
        buggy_code = remove_empty_lines(dp['buggy_solution'])
        if not correct_code.endswith("\n"):
            correct_code += "\n"

        if not buggy_code.endswith("\n"):
            buggy_code += "\n"

        correct_split_list =  utils.find_all_occurrences("\n", correct_code)
        error_split_list = utils.find_all_occurrences("\n", buggy_code)
        correct_lines = correct_code.split('\n')
        buggy_lines = buggy_code.split('\n')
        result = self.find_contrastive_index(correct_lines, buggy_lines)
        contrastive_pair = [[], []]
        for item in result:
            contrastive_pair[0].append(min(item[0], len(correct_split_list) - 1))
            contrastive_pair[1].append(min(item[1], len(error_split_list) - 1))
        line_label = [original_label] * len(error_split_list)
        for single in contrastive_pair[1]:
            line_label[single] = mutated_label
        item = {
            "context": "",
            "output": {
                "code_description": code_description,
                "problem_description": problem_description,
                "code": correct_code,
                "code_split_pos": correct_split_list
            },
            "mutated_code": {
                "code": buggy_code,
                "contrastive_pair": contrastive_pair,
                "code_split_pos": error_split_list,
                "line_label": line_label
            }
        }
        return item
    
class UniversalMutation:
    def __init__(self, cache_dir: str, mutator_path: str):
        self.cache_dir = cache_dir
        #self.pattern = r"PROCESSING MUTANT:\s*(\d+):\s*(.*?)\s*==>\s*(.*?)\.\.\.(VALID|INVALID|REDUNDANT)"
        #self.pattern = r"PROCESSING MUTANT:\s*(\d+):(.*?) ==> (.*?)\.\.\.(VALID|INVALID|REDUNDANT)"
        self.pattern = r"PROCESSING MUTANT:\s*(\d+): (.*?)  ==>  (.*?)\.\.\.(VALID|INVALID|REDUNDANT)"
        self.mutator_path = mutator_path # https://github.com/agroce/universalmutator/tree/master?tab=readme-ov-file
    
    def _mutate_code(self, code, line_number_list: list):
        """
        Mutate the given code according to the line_number_list.
        We anticpate the line_number_list is 0-based.
        """
        lines = "\n".join([str(i+1) for i in line_number_list]) # Turn to 1-based.
        code_file_name = "code.txt"
        line_file_name = "lines.txt"
        command = f"{self.mutator_path} {code_file_name} --lines {line_file_name} --showRules --noFastCheck"
        return_result = self.run_command_with_cache(command, code_file_name, line_file_name, code, lines)
        if "error" in return_result:
            logger.error(return_result["error"])
            raise Exception
        assert return_result['returncode'] == 0
        matches = re.findall(self.pattern, return_result["stdout"], re.DOTALL)
        result = dict()
        for match in matches:
            line_number, before, after, status = match
            line_number = int(line_number)
            before = before  # Clean up extra spaces in the "before" part
            after = after.split('...')[0]  # Capture only before '...' in the "after" part
            if status != "VALID":
                continue
            if (line_number - 1) not in result:
                result[line_number - 1] = [before, []]
            result[line_number - 1][1].append(after)
        return result
        
        
    def run_command_with_cache(self, command, code_file_name, line_file_name, code, lines):
        """
        Run the command in a shell with a temporary cache directory.
        Only return what has been left in the stdout.
        """
        try:
            assert not os.path.exists(self.cache_dir)
            # Step 1: Create a temporary directory (cache folder)
            os.makedirs(self.cache_dir)
            
            # Step 2: Create a file in the cache folder and write the content to it
            file_path = os.path.join(self.cache_dir, code_file_name)
            line_path = os.path.join(self.cache_dir, line_file_name)
            with open(file_path, 'w') as f:
                f.write(code)
                
            with open(line_path, 'w') as f:
                f.write(lines)

            # Step 3: Run the command in the cache directory
            process = subprocess.Popen(
                command, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                shell=True, 
                text=True, 
                cwd=self.cache_dir,
            )
            
            stdout, stderr = process.communicate()

            # Step 4: Record the output (stdout, stderr) and the return code
            result = {
                'stdout': stdout,
                'stderr': stderr,
                'returncode': process.returncode
            }

            return result

        except Exception as e:
            return {'error': str(e)}
        
        finally:
            # Step 5: Delete the cache folder once the command has finished executing
            if os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir)
            
def remove_empty_lines(input_string):
    # Split the string into lines
    lines = input_string.splitlines()
    # Filter out empty lines
    non_empty_lines = [line for line in lines if line.strip() != "" and line.strip().startswith("```") == False]
    # Join the non-empty lines back into a single string
    return "\n".join(non_empty_lines)

def select_line_according_to_split(split_token, code, previous_pos, cur_pos):
    real_start_pos = previous_pos + len(split_token) if previous_pos >= 0 else 0
    return code[real_start_pos: cur_pos+len(split_token)]
    
class AugPretrainCodedata(PretrainCodedata):
    def __init__(self,
                 mutate_prop: float, 
                 dataset_id: str,
                 feature_key_list: list=["java", "python", "c++", "javascript"],
                 preprocess_all_in_memory=False,
                 split_token="\n",
                 original_label=1,
                 mutated_label=2,
                 external_mutator=None,
                 meta_info_file_name = "meta.json",
                 data_file_name = "data.json",
                 load_from_disk = False,
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
        super().__init__(dataset_id, feature_key_list, preprocess_all_in_memory, split_token, load_from_disk)
        self.mutate_prop = mutate_prop
        self.original_label = original_label
        self.mutated_label = mutated_label
        self.external_mutator = external_mutator
        self.meta_info_file_name = meta_info_file_name
        self.data_file_name = data_file_name
        if preprocess_all_in_memory and not load_from_disk:
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
                    cur_code = self.switch_inside(code, code_split_pos, idx, select_line_according_to_split(self.split_token, code, last_split_pos, now_split_pos))
                elif op == "switch_outside":
                    cur_code = self.switch_outside(idx, select_line_according_to_split(self.split_token, code, last_split_pos, now_split_pos))
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
                cur_code = select_line_according_to_split(self.split_token, code, last_split_pos, now_split_pos)
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
        return new_mutated, new_mutated_code_split_pos, labels, [contrastive_original, contrastive_mutated]

    def aug_data_single(self, index):
        if self.preprocess_all_in_memory:
            item = self.processed_data[index]
            item_list = [item]
        else:
            item = self.data[index]
            item_list = self._preprocess_data_single(item) # Return List
        for item in item_list:
            if self.external_mutator is None:
                # Default Mode
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
            else:
                # Mutation mode: universal
                code = item['output']['code']
                split_pos = utils.find_all_occurrences(self.split_token, code)
                #code_split_pos = item['output']['code_split_pos']
                mutated_pos_list = []
                split_mapping = dict()
                previous_pos = 0
                for idx, pos in enumerate(split_pos):
                    if code[previous_pos:pos].strip() == "":
                        previous_pos = pos
                        split_mapping[idx] = None
                        continue
                    else:
                        mutated_pos_list.append(idx) # record which line should be the candidate to be mutated
                        split_mapping[idx] = len(mutated_pos_list) - 1
                        previous_pos = pos
                code_split_pos_len = len(mutated_pos_list)
                random.shuffle(mutated_pos_list)
                mutated_num = max(math.ceil(code_split_pos_len * self.mutate_prop), 1)
                local_pointer = 0
                remaining_num = mutated_num
                new_mutated_info = dict()

                naive_flag = [False, False] # A hacky way to deal with continue and break rules
                while local_pointer < code_split_pos_len:
                    # Iteratively mutate the code
                    # Until mutated_num is reached
                    candidate_pos_list = mutated_pos_list[local_pointer: local_pointer + remaining_num]
                    new_info = self.external_mutator._mutate_code(code, candidate_pos_list)
                    clean_key = list(new_info.keys())
                    for key in clean_key:
                        # Here we try to avoid the cases
                        # where most mutated code lines are only generated
                        # through continue/break style.
                        # Refer to https://github.com/agroce/universalmutator/blob/55c68f7377c15f08292cd148e8e46dd029704bec/universalmutator/static/universal.rules#L83
                        temp = []
                        for _ in new_info[key][1]:
                            if "continue" not in new_info[key][0] and "continue" in _:
                                if naive_flag[0] is False:
                                    temp.append(_)
                                    naive_flag[0] = True
                                else:
                                    continue
                            elif "break" not in new_info[key][0] and "break" in _:
                                if naive_flag[1] is False:
                                    temp.append(_)
                                    naive_flag[1] = True
                                else:
                                    continue
                            else:
                                temp.append(_)
                        if len(temp) == 0:
                            new_info.pop(key)
                        else:
                            new_info[key][1] = temp
                    new_mutated_info.update(new_info)
                    if mutated_num == len(new_mutated_info):
                        break
                    remaining_num = mutated_num - len(new_mutated_info)
                    local_pointer += len(candidate_pos_list)

                    
                code_in_line = code.split("\n")
                new_code_in_line = []
                labels = []
                #new_split_mapping = []
                new_mutated_code_split_pos = []
                contrastive_original = []
                contrastive_mutated = []

                pos_in_mutated_code = 0

                for idx, line in enumerate(code_in_line):
                    if idx in new_mutated_info:
                        # Get the replacement code (handle multiple options)
                        replacement_options = new_mutated_info[idx][1]
                        new_code = random.choice(replacement_options)
                        new_code_lines = new_code.split('\n')  # Split into multiple lines if necessary

                        # Extend the new code lines into the new_code_in_line
                        new_code_in_line.extend(new_code_lines)

                        # Update labels and split_mapping for the new lines
                        for inner_idx, _ in enumerate(new_code_lines):
                            if line == _:
                                labels.append(self.original_label)
                            else:
                                labels.append(self.mutated_label)
                                # Update contrastive mappings
                                if split_mapping[idx] is not None:
                                    contrastive_original.append(split_mapping[idx])
                                    contrastive_mutated.append(len(labels) - 1)
                            #new_split_mapping.append(None)  # No direct mapping for new lines

                            # Update the position in the mutated code
                            pos_in_mutated_code += len(new_code_lines[inner_idx]) + 1  # +1 for '\n'
                            new_mutated_code_split_pos.append(pos_in_mutated_code - 1)

                    else:
                        # No mutation; append the original line
                        new_code_in_line.append(line)
                        # Update the position in the mutated code
                        pos_in_mutated_code += len(line) + 1  # +1 for '\n'
                        # We do not record label and split_pos for empty line
                        if len(line.strip()) == 0:
                            continue
                        labels.append(self.original_label) # if split_mapping[idx] is not None else None)
                        #new_split_mapping.append(split_mapping[idx])

                        new_mutated_code_split_pos.append(pos_in_mutated_code - 1)
                new_mutated = "\n".join(new_code_in_line)
                contrastive_pair = [contrastive_original, contrastive_mutated]
                mutated_op_list = None
                
                
            assert len(contrastive_pair[0]) == len(contrastive_pair[1])
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
            select_line = select_line_according_to_split(self.split_token, cur_code, start_pos, end_pos)
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
                select_line = select_line_according_to_split(self.split_token, cur_code, start_pos, end_pos)
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

    @classmethod
    def load(cls, path, meta_info_file_name="meta.json"):
        with open(os.path.join(path, meta_info_file_name), "r") as f:
            meta_info = json.load(f)
        this_dataset = cls(
            meta_info["mutate_prop"],
            meta_info["dataset_id"],
            meta_info["feature_key_list"],
            meta_info["preprocess_all_in_memory"],
            meta_info["split_token"],
            meta_info["original_label"],
            meta_info["mutated_label"],
            meta_info_file_name=meta_info_file_name,
            data_file_name=meta_info["data_file_name"],
            load_from_disk=True
        )
        with open(os.path.join(path, meta_info["data_file_name"]), "r") as f:
            data = json.load(f)
        
        this_dataset.processed_data = data["processed_data"]
        this_dataset.data = datasets.arrow_dataset.Dataset.from_dict(data["data"])
        return this_dataset
        
    
    def save_to_disk(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        meta_info_path = os.path.join(path, self.meta_info_file_name)
        meta_info = {
            "dataset_id": self.dataset_id,
            "split_token": self.split_token,
            "feature_key_list": self.feature_key_list,
            "preprocess_all_in_memory": self.preprocess_all_in_memory,
            "mutate_prop": self.mutate_prop,
            "original_label": self.original_label,
            "mutated_label": self.mutated_label,
            "meta_info_file_name": self.meta_info_file_name,
            "data_file_name": self.data_file_name
        }
        with open(meta_info_path, "w") as f:
            json.dump(meta_info, f)
        
        with open(os.path.join(path, self.data_file_name), "w") as f:
            json.dump(
                {
                    "processed_data": self.processed_data,
                    "data": self.data.to_dict()
                },
                f
            )
        
    
class AugExecCodeData(Dataset):
    def __init__(self, 
                 data_path, 
                 tokenizer,
                 split_token="\n",
                 additional_prompt="", 
                 max_length=2048,
                 original_label=1,
                 mutated_label=2):
        self.data_path = data_path
        self.original_label = original_label
        self.mutated_label = mutated_label
        self.raw_data = pd.read_csv(data_path, index_col=0)
        self.error_code = self.raw_data['sourceText']
        self.correct_code = self.raw_data["targetText"]
        self.sourceLineText = self.raw_data["sourceLineText"]
        self.targetLineText = self.raw_data["targetLineText"]
        self.diffText_del = self.raw_data["diffText_del"]
        self.error_linenum = [i - 1 for i in self.raw_data['lineNums_Abs']] # From 1-based to 0-based
        self.correct_linenum = [i - 1 for i in self.raw_data['lineNums_Abs']] # From 1-based to 0-based
        self.error_info = self.raw_data['sourceErrorClangParse']
        self.tokenizer = tokenizer
        self.split_token = split_token
        self.token_output = []
        self.additional_prompt = additional_prompt
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
            if inputs["input_ids"].shape[-1] >= max_length:
                drop_list.append(idx)
            self.token_output.append(input_ids)
        
        self.error_code = [value for i, value in enumerate(self.error_code) if i not in drop_list]
        self.correct_code = [value for i, value in enumerate(self.correct_code) if i not in drop_list]
        self.error_linenum = [value for i, value in enumerate(self.error_linenum) if i not in drop_list]
        self.error_info = [value for i, value in enumerate(self.error_info) if i not in drop_list]
        self.token_output = [value for i, value in enumerate(self.token_output) if i not in drop_list]
        self.sourceLineText = [value for i, value in enumerate(self.sourceLineText) if i not in drop_list]
        self.targetLineText = [value for i, value in enumerate(self.targetLineText) if i not in drop_list]
        self.diffText_del = [value for i, value in enumerate(self.diffText_del) if i not in drop_list]
        self.correct_linenum = [value for i, value in enumerate(self.correct_linenum) if i not in drop_list]
        self.raw_data = self.raw_data.drop(index=drop_list).reset_index(drop=True)

        self.error_split_pos = []
        self.correct_split_pos = []
        self.labels = []
        drop_list = []
        for idx in range(len(self.error_code)):
            code = self.error_code[idx]
            processed_split_pos, remove_index = self._obtain_split_pos(code)
            self.error_split_pos.append(processed_split_pos)
            error_linenums = adjust_indices([self.error_linenum[idx]], remove_index)
            if len(error_linenums) < 1 or len(processed_split_pos) <= error_linenums[0]:
                drop_list.append(idx)
                self.error_linenum[idx] = -1
            else:
                self.error_linenum[idx] = error_linenums[0]
            correct_code = self.correct_code[idx]
            processed_split_pos, remove_index = self._obtain_split_pos(correct_code)
            self.correct_split_pos.append(processed_split_pos)
            correct_linenums = adjust_indices([self.correct_linenum[idx]], remove_index)
            if len(correct_linenums) < 1 or len(processed_split_pos) <= correct_linenums[0]:
                drop_list.append(idx)
                self.correct_linenum[idx] = -1
            else:
                self.correct_linenum[idx] = correct_linenums[0]
            
            
            if len(drop_list) > 0 and drop_list[-1] != idx:
                end_pos = self.error_split_pos[idx][self.error_linenum[idx]]
                prev_end_pos = self.error_split_pos[idx][self.error_linenum[idx]-1] if self.error_linenum[idx] > 0 else -1
                error_code_line = select_line_according_to_split("\n", self.error_code[idx], prev_end_pos, end_pos)
                end_pos = self.correct_split_pos[idx][self.correct_linenum[idx]]
                prev_end_pos = self.correct_split_pos[idx][self.correct_linenum[idx]-1] if self.correct_linenum[idx] > 0 else -1
                correct_code_line = select_line_according_to_split("\n", self.correct_code[idx], prev_end_pos, end_pos)
                if error_code_line.strip() == correct_code_line.strip():
                    drop_list.append(idx)
            
            
            #if self.diffText_del[idx] is not np.nan:
                #temp = utils.find_all_occurrences("\n", self.diffText_del[idx])
            #    self.correct_linenum[idx] += 1
            #self.correct_split_pos.append(processed_split_pos)
        

        self.error_code = [value for i, value in enumerate(self.error_code) if i not in drop_list]
        self.correct_code = [value for i, value in enumerate(self.correct_code) if i not in drop_list]
        self.error_linenum = [value for i, value in enumerate(self.error_linenum) if i not in drop_list]
        self.error_info = [value for i, value in enumerate(self.error_info) if i not in drop_list]
        self.token_output = [value for i, value in enumerate(self.token_output) if i not in drop_list]
        self.sourceLineText = [value for i, value in enumerate(self.sourceLineText) if i not in drop_list]
        self.targetLineText = [value for i, value in enumerate(self.targetLineText) if i not in drop_list]
        self.raw_data = self.raw_data.drop(index=drop_list).reset_index(drop=True)
        self.error_split_pos = [value for i, value in enumerate(self.error_split_pos) if i not in drop_list]
        self.correct_split_pos = [value for i, value in enumerate(self.correct_split_pos) if i not in drop_list]
        self.diffText_del = [value for i, value in enumerate(self.diffText_del) if i not in drop_list]
        self.correct_linenum = [value for i, value in enumerate(self.correct_linenum) if i not in drop_list]
        for idx, item in enumerate(self.error_split_pos):
            line_label = [self.original_label] * len(item)
            temp_idx = self.error_linenum[idx]
            line_label[temp_idx] = self.mutated_label
            self.labels.append(line_label)

    
    def _obtain_split_pos(self, code):
        split_pos = utils.find_all_occurrences(self.split_token, code)
        previous_pos = 0
        processed_split_pos = []
        remove_index = []
        for idx, pos in enumerate(split_pos):
            if code[previous_pos:pos].strip() == "":
                previous_pos = pos
                remove_index.append(idx)
                continue
            else:
                processed_split_pos.append(pos)
                previous_pos = pos
        return processed_split_pos, remove_index
    
    def __getitem__(self, idx):
        return_data = {
            "context": self.additional_prompt,
            "output": {
                "code_description": "",
                "problem_description": "",
                "code": self.correct_code[idx],
                "code_split_pos": self.correct_split_pos[idx],
            },
            "mutated_code": {
                "code": self.error_code[idx],
                "contrastive_pair": [[self.correct_linenum[idx]], [self.error_linenum[idx]]],
                "code_split_pos": self.error_split_pos[idx],
                "line_label": self.labels[idx],
            },
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

def wrap_input(item, code, code_split_pos, tokenizer, split_token, contrastive_list):
    
    context = item['context']
    code_description = item['output']['code_description']
    problem_description = item['output']['problem_description']
    description = problem_description + code_description
    target_input = context + description + f"{split_token}```{split_token}" + code
    context_str_len = len(context + description + f"{split_token}```{split_token}")
    input_info = tokenizer(target_input, return_tensors="pt", return_offsets_mapping=True)
    offset_mapping = input_info["offset_mapping"].squeeze().tolist()
    input_info.pop("offset_mapping")
    split_tok_pos = list()
    removal_indices = list()
    start_token_idx = utils.match_token_in_offset_mapping(offset_mapping, context_str_len, context_str_len+len(split_token))[0]
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
    return input_info, split_tok_pos, contrastive_list, start_token_idx
    
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
        input_info, original_split_tok_pos, contrastive_list_original, start_token_idx = wrap_input(item, 
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
            hidden_states = hidden_states[0] # (token_num, hidden_size)
            end_token_idx = len(hidden_states) - 1
            start_hidden = hidden_states[start_token_idx, :]
            last_hidden = hidden_states[end_token_idx, :]
            hidden_states = hidden_states[original_split_tok_pos, :] # (split_token_num, hidden_size)
            recorder.clear_cache()
            
            store_info = {
                "original": {"hidden_states": hidden_states,
                             "start_hidden": start_hidden, 
                             "last_hidden": last_hidden, 
                             "start_in": start_token_idx in original_split_tok_pos,
                             "last_in": end_token_idx in original_split_tok_pos,},
                "original_split_tok_pos": original_split_tok_pos,   
            }
        else:
            store_info = store[idx]
        if "mutated_code" not in store_info:
            store_info["mutated_code"] = {}
        
        input_info, mutated_split_tok_pos, contrastive_list_mutated, start_token_idx = wrap_input(item,
                                                                                 item['mutated_code']['code'],
                                                                                 item['mutated_code']['code_split_pos'],
                                                                                 tokenizer,
                                                                                 dataset.split_token,
                                                                                 item['mutated_code']['contrastive_pair'][1]
                                                                                 )
        _, record_dict = recorder.forward(input_info)
        
        hidden_states = record_dict[list(recorder.layer_names.keys())[0]] # Only one layer
        hidden_states = hidden_states[0] # (1, token_num, hidden_size)
        hidden_states = hidden_states[0] # (token_num, hidden_size)
        end_token_idx = len(hidden_states) - 1
        start_hidden = hidden_states[start_token_idx, :]
        last_hidden = hidden_states[end_token_idx, :]
        hidden_states = hidden_states[mutated_split_tok_pos, :] # (split_token_num, hidden_size)
        
        key_list = [int(i) for i in list(store_info["mutated_code"].keys())]
        if len(key_list) == 0:
            cur_key = str(0)
        else:
            cur_key = max(key_list) + 1
            cur_key = str(cur_key)
        store_info["mutated_code"][cur_key] = {"hidden_states": hidden_states, 
                                               "contrastive_list_original": contrastive_list_original,
                                               "contrastive_list_mutated": contrastive_list_mutated, 
                                               "line_label": item['mutated_code']['line_label'],
                                                "start_hidden": start_hidden,
                                                "last_hidden": last_hidden,
                                                "start_in": start_token_idx in mutated_split_tok_pos,
                                                "last_in": end_token_idx in mutated_split_tok_pos,
                                               }
        if run_original:
            store.append(store_info)
        else:
            store.set_item(idx, store_info)
    
    store.save_to_disk()
            
            
        
# python3 method/collect_hidden_aug.py --config-file model_eval_config/CodeLlama_hidden_state.yaml --result-output-path /data/data_disk/trust_code/leetcode_aug3_fix
# python3 method/collect_hidden_aug.py --config-file model_eval_config/CodeLlama_hidden_state_tracer.yaml --result-output-path /data/data_disk/trust_code/tracer
#app = typer.Typer(pretty_exceptions_show_locals=False, pretty_exceptions_short=False)
app = typer.Typer(pretty_exceptions_short=False)
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
    mutation_prop = config_dict['task_config'].get("mutation_prop", None)
    feature_key_list = config_dict['task_config']["feature_key_list"]
    dataset_path_list = config_dict['task_config'].get("dataset_path_list", None)
    
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

    if "tracer" in dataset_name:
        aug_times = 0
        logger.info("Using AugExecCodeData for Tracer dataset")
        data = AugExecCodeData(
            data_path=dataset_name,
            tokenizer=tokenizer,
        )
    elif "humanevalpack" in dataset_name:
        aug_times = 0
        logger.info("Using HumanEvalPack as dataset")
        data = HumanEvalPack(dataset_name=dataset_name, feature_key_list=feature_key_list)
    else:
        aug_times = len(mutation_prop)
        if dataset_path_list is not None:
            data = AugPretrainCodedata.load(dataset_path_list[0])
            aug_times = len(dataset_path_list)
        else:
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
        if dataset_path_list is not None:
            data = AugPretrainCodedata.load(dataset_path_list[i])
        else:
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
    