import os
import subprocess
import shutil
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from collections import OrderedDict
import torch
import json
import shelve
from pygments.lexers import get_lexer_by_name
from pygments.token import Token
import numpy as np
import difflib
from loguru import logger
import re

HARD_TOKEN_LIMIT = 4096 - 512
CODE_NOT_FOUND_FLAG = "NO_CODE"

def match_token_in_offset_mapping(offset_mapping, start_pos, end_pos):
    """
    Match the target_str in the tokenized_info["offset_mapping"].
    Return the start index and the end index of the `token list`
    """
    
    start_idx = None
    end_idx = None
    for idx, (left, right) in enumerate(offset_mapping):
        if start_idx is None and (start_pos >= left and start_pos < right):
            start_idx = idx
        if end_idx is None and (end_pos >= left and end_pos <= right):
            end_idx = idx
            break
    assert start_idx is not None
    assert end_idx is not None
    
    return (start_idx, end_idx)

def identify_start_end_substr(original_tokens, target_str, tokenizer):
    """
    Find the start index and end index of the target_str in the original_str for tokenized string.
    Return the start index and the end index of the `token list`
    """
    original_text = tokenizer.decode(original_tokens, skip_special_tokens=True)
    #original_text = tokenizer.decode(original_tokens, skip_special_tokens=False)
    original_token_info = tokenizer(original_text, return_offsets_mapping=True) # dict_keys(['input_ids', 'attention_mask', 'offset_mapping'])
    start_pos = original_text.find(target_str)
    assert start_pos > 0, "target_str not in original_tokens"
    end_pos = start_pos + len(target_str)
    
    result = match_token_in_offset_mapping(original_token_info["offset_mapping"], start_pos, end_pos)
    
    return result

def aggregate_scores(token_list, scores, sep=["\n", "\n\n", "\n\n\n"]):
    result = [[], []]
    aggregate_token = list()
    aggregate_score = list()
    for idx, item in enumerate(token_list):
        if item in sep:
            if len(aggregate_token) == 0:
                continue
            result[0].append(aggregate_token)
            result[1].append(aggregate_score)
            aggregate_token = list()
            aggregate_score = list()
        else:
            aggregate_token.append(item)
            aggregate_score.append(scores[idx])
    return result

def postprocess_record(record, stop_sequences, tokenizer):
    """
    Post process the result.
    """
    def remove_stop_sequences(text, stop_seqs):
        #text = text.strip()
        for stop_seq in stop_seqs:
            if text.endswith(stop_seq):
                text = text[: -len(stop_seq)]
        return text
    
    str_all = record["str_all"]
    str_output = record["str_output"]
    input_length = record["input_length"]
    # Clean both str_all and str_output
    cleaned_str_all = remove_stop_sequences(str_all, stop_sequences)
    cleaned_str_output = remove_stop_sequences(str_output, stop_sequences)
    
    # Re-tokenize the cleaned_str_output to see how many tokens remain
    final_token_ids = tokenizer.encode(cleaned_str_output, add_special_tokens=False)
    final_length = len(final_token_ids)
    original_length = record["output_len"]
    tokens_to_remove = original_length - final_length
    
    if tokens_to_remove > 0:
        # Adjust gen_probs if it exists
        gen_probs = record["gen_probs"]
        gen_probs[0] = gen_probs[0][:final_length]
        record["gen_probs"] = gen_probs
        record["token_output"] = record["token_output"][:input_length + final_length]
        record["str_all"] = cleaned_str_all
        record["str_output"] = cleaned_str_output
        record["output_len"] = final_length
        
    return record

def generate_and_record(llm, tokenizer, input_str, generate_config={"max_new_tokens": 600}, extra_generation_config=None, remove_seq_list=None):
    # https://huggingface.co/docs/transformers/v4.44.2/en/main_classes/text_generation#transformers.GenerationMixin.generate
    # https://huggingface.co/docs/transformers/v4.44.2/en/main_classes/text_generation#transformers.GenerationConfig
    max_length = min(tokenizer.model_max_length, HARD_TOKEN_LIMIT)
    inputs = tokenizer(input_str, return_tensors="pt", max_length=max_length, truncation=True).to(llm.device)
    input_length = inputs["input_ids"].shape[-1]
    result = {"input_length": input_length}
    if "return_dict_in_generate" in generate_config:
        generate_config["output_logits"] = True
    with torch.inference_mode():
        output = llm.generate(
            **inputs,
            pad_token_id=tokenizer.eos_token_id,
            #tokenizer=tokenizer,
            **generate_config,
            generation_config=extra_generation_config
        )
    if "return_dict_in_generate" in generate_config:
        seq = output["sequences"].cpu()
        probs = torch.stack(output["logits"], dim=1).float().cpu().softmax(-1)
        gen_probs = torch.gather(probs, 2, seq[:, input_length:, None]).squeeze(-1)
        result["gen_probs"] = gen_probs.tolist()
    else:
        seq = output.to("cpu")  # Move output back to CPU
        
    seq = seq[0]
    str_all = tokenizer.decode(seq)
    str_output = tokenizer.decode(seq[input_length:], skip_special_tokens=True)
    result["output_len"] = seq[input_length:].shape[0]
    result["str_all"] = str_all
    result["str_output"] = str_output
    token_output = seq.tolist()
    result["token_output"] = token_output
    if remove_seq_list is not None:
        result = postprocess_record(result, remove_seq_list, tokenizer)
    return result



def load_tokenizer(model_name, model_path=None):
    assert model_name in ["Phind/Phind-CodeLlama-34B-v2", "Qwen/Qwen2.5-Coder-32B", "bigcode/starcoder2-15b", "deepseek-ai/deepseek-coder-33b-base"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

def load_opensource_model(model_name, 
                          parallel=True, 
                          quantization=None, 
                          model_path=None, 
                          cache_dir=None, 
                          user_args=None, 
                          device_map=None):
    if parallel:
        args = {"device_map": "balanced"}
    else:
        args = {"device_map": "cpu"}
    if device_map is not None:
        args["device_map"] = device_map
    if user_args is not None:
        args.update(user_args)
    if cache_dir is not None:
        args["cache_dir"] = cache_dir
    assert quantization in [None, "8bit", "4bit", "16bit"]
    if quantization == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            #load_in_8bit=True,
            #bnb_8bit_compute_dtype=torch.float16
        )
    elif quantization == "8bit":
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    elif quantization == "16bit":
        args["torch_dtype"] = torch.float16 #torch.bfloat16
        quantization_config = None
    else:
        quantization_config = None
    
    tokenizer = load_tokenizer(model_name)
    assert model_name in ["Phind/Phind-CodeLlama-34B-v2", "Qwen/Qwen2.5-Coder-32B", "bigcode/starcoder2-15b", "deepseek-ai/deepseek-coder-33b-base"]
    model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                **args
            )
    return (model, tokenizer)

def write_jsonl(output_path, data):
    with open(output_path, "w") as ifile:
        for entry in data:
            json_line = json.dumps(entry)
            ifile.write(json_line + '\n')

def read_jsonl(file_path):
    """
    Reads a JSONL file and returns a list of JSON objects (dictionaries).
    
    :param file_path: Path to the JSONL file to read.
    :return: A list of dictionaries, where each dictionary is a JSON object from the file.
    """
    data = []  # Initialize an empty list to store the JSON objects
    
    # Open the JSONL file in read mode
    with open(file_path, 'r') as file:
        # Iterate over each line in the file
        for line in file:
            # Strip whitespace and parse the line as a JSON object
            json_object = json.loads(line)
            # Append the parsed JSON object (dictionary) to the list
            data.append(json_object)
    
    return data
    
def get_token_indices(mapping, start_char_pos, end_char_pos):
    """
    Obtain the token positions according to the mappping.
    ===
    Args:
        mapping: dict, returned by return_offsets_mapping.
        start_char_pos: int,
        end_char_pos: int,
    Return:
        List[int]: 
    """
    indices = []
    for idx, (token_start, token_end) in enumerate(mapping):
        # Check if there's any overlap between the token and the target substring
        if (token_end > start_char_pos and token_start < end_char_pos):
            indices.append(idx)
    return indices

def get_line_to_char_index_mapping(text):
    """
    Obtain the string split by lines --> original character index.
    Return:
        List[List]: the list of [start, end) line position.
    """
    lines = text.splitlines(keepends=True)  # Split lines but keep newline characters
    index_mapping = []
    current_index = 0
    
    for line in lines:
        line_length = len(line)
        line_mapping = [current_index, current_index + line_length]
        index_mapping.append(line_mapping)
        current_index += line_length
    
    return index_mapping


def get_important_token_pos(important_line_info, data, tokenizer):
    """
    Get the position idx of the important token given the important_line_info.
    ---
    Args:
        important_line_info: dict. data_index -> line mapping (List[List]) List of important target lines
        data: dict. data_index -> collected data mapping
        tokenizer: TokenizerFast. The tokenizer used in model inference.
    Output:
        target_buggy_positions: dict. data_index -> List[List]. Mapping from data_index to list of [start, end) token positions
        important_token_info: dict. data_index -> List[List]. Mapping from data_index to the idx of token idx.
            Each list corresponds to a ``` ``` block in the OPENAI response.
    """
    target_buggy_positions = OrderedDict()
    important_token_info = OrderedDict()
    for key in important_line_info:
        split_response = data[key]['str_output'].splitlines()
        single_info = important_line_info[key]
        line_char_mapping = get_line_to_char_index_mapping(data[key]['str_output'])
        target_buggy_positions[key] = list()
        important_token_info[key] = list()
        #tokenized_info = tokenizer(data[key]['str_output'], return_offsets_mapping=True, add_special_tokens=False)
        #tokenized_info_start_idx = remove_redundant_tuples(tokenized_info["offset_mapping"])
        #offset_mapping = tokenized_info["offset_mapping"][tokenized_info_start_idx:]
        tokenized_info = ModelOutputCleaner.clean_first_tokenization(data[key]['str_output'], tokenizer)
        for item in single_info:
            #start_line_number = item[0]
            #end_line_number = item[1]
            cleaned_line_number = list()
            for inner_line_idx in item:
                if is_line_only_punctuators_pygments(split_response[inner_line_idx]):
                    continue
                else:
                    cleaned_line_number.append(inner_line_idx)
            
            mapping_result = [line_char_mapping[i] for i in cleaned_line_number]
            target_buggy_positions[key] += mapping_result
            #important_token_info[key].append([])
            for single in mapping_result:
                important_token_info[key].append(get_token_indices(tokenized_info["offset_mapping"], single[0], single[1]))
    return (target_buggy_positions, important_token_info)


def is_line_only_punctuators_pygments(line, language='c'):
    """
    Return True if the provided code lins only contains punctuators.
    """
    lexer = get_lexer_by_name(language)
    tokens = lexer.get_tokens(line)
    for tok_type, tok_val in tokens:
        if tok_type in Token.Text.Whitespace:
            continue
        # Check if token is a punctuation or operator
        elif tok_type in Token.Punctuation:
            continue
        elif tok_type in Token.Comment:
            return True # Comment Line. Skip
        else:
            # Any other token type means the line contains more than punctuators
            return False
    return True

def obtain_topk_tokens_by_prob(data, candidate_tokens, key, topk, agg_method=np.mean):
    """
    Obtain topk line tokens according to probability indicators.
    """
    # temp = np.array(data[key]['token_output'][data[key]['input_length']:])
    gen_probes = np.array(data[key]['gen_probs'][0])
    result = []
    for per_line in candidate_tokens:
        result.append(gen_probes[per_line])
    agg_result = [agg_method(i) for i in result]
    rank_per_line = sorted(list(zip(agg_result, [i for i in range(len(agg_result))]))) # [:topk] # From low to high
    rank_per_line = [i[1] for i in rank_per_line]
    #selected_token = set()
    #for line in rank_per_line:
    #    selected_token = selected_token.union(set(candidate_tokens[line[1]]))
    return agg_result, rank_per_line

def load_shelve(path):
    path = str(path)
    with shelve.open(path) as db:
        loaded_data = dict(db)
    return loaded_data

def load_shelve_and_resume(dir_path):
    """
    Looping through the files under the dir_path. If there is more than
    one shelve file, raise an Exception. Otherwise, record the already
    saved count using len(loaded_data) and return the next index
    to resume.
    """
    dir_path = str(dir_path)
    # Get list of all shelve files in the directory
    shelve_files = [f for f in os.listdir(dir_path) if f.endswith('.db') or f.endswith('.dat')]

    # If there's more than one shelve file, raise an exception
    if len(shelve_files) > 1:
        raise Exception("More than one shelve file found in the directory.")
    
    # If no shelve file is found, return 0 as the starting index
    if not shelve_files:
        return 0, None
    
    # Load the shelve file
    shelve_file = os.path.splitext(shelve_files[0])[0]
    path = os.path.join(dir_path, shelve_file)
    with shelve.open(path) as db:
        loaded_data = list(db.keys())
    
    # Return the next index to resume
    return len(loaded_data), path

def find_all_occurrences(substring, string):
    """
    Find all occurrences of a substring in a string.
    """
    positions = []
    start = 0
    while start < len(string):
        pos = string.find(substring, start)
        if pos == -1:
            break
        positions.append(pos)
        start = pos + 1
    return positions

def extract_code_block(text, select_idx=None):
    """
    Extract code block enclosed by ``` ```.
    Return the code block and its start and end positions in the text.
    """
    code_blocks = []
    code_blocks_info = []
    start = 0
    
    code_start_shift = 0
    text_length = len(text)
    if len(re.findall("```", text)) == 1:
        if not text.startswith("```"):
            text = "```\n" + text
            code_start_shift = 4
        elif not text.strip().endswith("```"):
            text = text +  "\n```"

    while True:
        # Find the start of a code block
        start_idx = text.find("```", start)
        if start_idx == -1:
            break

        # Find the end of the code block, starting after the found start
        end_idx = text.find("```", start_idx + 3)
        if end_idx == -1:
            break

        # Extract the content between the backticks
        # Skip any text on the same line as the opening backticks
        code_start = text.find("\n", start_idx + 3) + 1  # Find the start of the next line after the opening ```
        #print(code_start)
        #print(end_idx)
        code_block = text[code_start:end_idx]#.strip()
        

        # Update the start position to search for the next code block
        start = end_idx
        if len(code_block) == 0:
            continue
        
        code_blocks.append(code_block)
        code_start = max(0, code_start - code_start_shift)
        end_idx = end_idx - code_start_shift
        end_idx = min(end_idx, text_length)
        code_blocks_info.append([code_start, end_idx])

    if len(code_blocks) == 0:
        return None, None
    
    if select_idx is None:
        return code_blocks, code_blocks_info  # Return all matches if no specific index is requested

    # Return specific match if select_idx is provided and within range
    if select_idx < len(code_blocks):
        return code_blocks[select_idx], code_blocks_info[select_idx]
    else:
        return None, None  # Return None if select_idx is out of bounds
    


    


def label_seg(gt_seg, output_seg):
    """
    Label segmentation as 1 if any tokens are presented in gt_seg.
    ---
    Args:
        gt_seg: List[List]
        output_seg: List[List]
    Output:
        List[Bool]
    """
    result = [0 for i in range(len(output_seg))]
    g_set = []
    for seg in gt_seg:
        g_set += seg
    g_set = set(g_set)
    for idx, o_seg in enumerate(output_seg):
        if len(set(o_seg).intersection(g_set)) > 0:
            result[idx] = 1
    return result



def normalize_code(code, language):
    """
    Remove trailing comments from code lines.
    ---
    Args:
        code: string. Raw code string.
        language: string. Type of programming language.
    Return:
        string. Code string without comments.
    """
    
    lexer = get_lexer_by_name(language)
    tokens = lexer.get_tokens(code)
    # When the whole line is a comment
    # This filter will leave one single \n
    filtered_code = []
    for token_type, token_value in tokens:
        if token_type not in Token.Comment:
            filtered_code.append(token_value)
        else:
            for i in range(len(token_value.splitlines())-1):
                filtered_code.append("\n")
    #filtered_code = ''.join(token_value for token_type, token_value in tokens if token_type not in Token.Comment)
    filtered_code = "".join(filtered_code)
    filtered_code = filtered_code.splitlines(keepends=True)
    result = list()
    for line in filtered_code:
        if is_line_only_punctuators_pygments(line, language):
            result.append("")
        else:
            result.append(line.strip())
    return result

# Function to identify changes and record positions in code_b
def get_changes_with_line_numbers(gt_code, generated_code, language):
    '''
    # Example usage
    code_a = """def foo():
        print("Hello")
        print("hi")
        #....
        return 1
    """

    code_b = """def foo(s):
        #....
        print("Hello")
        #....
        return 1
    """
    Return --> [Line 0], [Line 0, Line 4] --> [Line 0, Line 4]
    '''
    code_a_lines = normalize_code(gt_code, language)
    code_b_lines = normalize_code(generated_code, language)

    # Use ndiff to compare and track changes
    diff = list(difflib.ndiff(code_a_lines, code_b_lines))

    changes_in_b = []  # Changes recorded with line numbers in code_b
    missing_from_b = []  # Lines deleted from code_a and their expected positions in code_b
    #a_line_num = 0  # Line number tracker for code_a
    b_line_num = 0  # Line number tracker for code_b
    for i, line in enumerate(diff):
        if line.startswith(' '):  # Unchanged lines
            #a_line_num += 1
            b_line_num += 1
        elif line.startswith('-'):  # Deletion from code_a (missing in code_b)
            # The current b_line_num is empty, skip it
            if len(line[2:]) == 0:
                continue
            break_flag = False
            if b_line_num >= len(code_b_lines):
                b_line_num = len(code_b_lines) - 1
                break_flag = True
            b_line_num_shifted = b_line_num
            if len(code_b_lines[b_line_num]) == 0:
                temp_continue_flag = False
                for j in range(b_line_num+1, len(code_b_lines)):
                    b_line_num_shifted = j
                    if len(code_b_lines[j]) > 0:
                        missing_from_b.append((b_line_num_shifted, line[2:]))
                        temp_continue_flag = True  
                        break
                if (not temp_continue_flag) and (b_line_num_shifted + 1 >= len(code_b_lines)):
                    # Search from the bottom of the code_b to find the last non-empty line
                    for j in range(b_line_num_shifted, 0, -1):
                        if len(code_b_lines[j]) > 0:
                            b_line_num_shifted = j
                            missing_from_b.append((b_line_num_shifted, line[2:]))  
                            break
                    break
                if break_flag == True:
                    break
            else:
                missing_from_b.append((b_line_num, line[2:]))  # Record 0-based expected line number in code_b
                #a_line_num += 1
        elif line.startswith('+'):  # Addition in code_b
            if len(code_b_lines[b_line_num]) == 0:
                # Addition in Comment type is ignored.
                #print(line, b_line_num)
                b_line_num += 1
            else:
                #print(line, b_line_num)
                changes_in_b.append((b_line_num, line[2:]))  # Record 0-based line number in code_b
                b_line_num += 1
        #if b_line_num >= len(code_b_lines):
        #    break

    return changes_in_b, missing_from_b

class ModelOutputCleaner:
    def __init__(self, start_token, model_type):
        self.start_token = start_token
        self.model_type = model_type
    
    def forward(self, str_output):
        if "llama" in self.model_type.lower():
            # We only need to deal with start token
            # Bcause this class is used for cleanning model output
            # for one single inference, which is used to
            # extract hidden states and attention scores.
            return str_output.replace(self.start_token, "")
        else:
            raise NotImplementedError
    
    @staticmethod
    def clean_first_tokenization(str_output, tokenizer):
        def remove_redundant_tuples(input_list):
            # Check how many initial (0, 1) tuples are present
            # This is used to deal with https://github.com/huggingface/transformers/issues/26273
            count = 0
            for item in input_list:
                if item == (0, 1):
                    count += 1
                else:
                    break

            # Remove redundant (0, 1) tuples, keep only one if there are multiple
            start_idx = max(count - 1, 0)
            #if count > 1:
            #    input_list = input_list[count - 1:]

            return start_idx
        
        tokenized_info = tokenizer(str_output, return_offsets_mapping=True, add_special_tokens=False)
        tokenized_info_start_idx = remove_redundant_tuples(tokenized_info["offset_mapping"])
        offset_mapping = tokenized_info["offset_mapping"][tokenized_info_start_idx:]
        input_ids = tokenized_info["input_ids"][tokenized_info_start_idx:]
        attention_mask = tokenized_info["attention_mask"][tokenized_info_start_idx:]
        result = {"input_ids": input_ids, "attention_mask": attention_mask, "offset_mapping": offset_mapping}
        return result


import ast

def extract_top_level_function_names(code):
    function_names = []
    tree = ast.parse(code)
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef):
            function_names.append(node.name)
    return function_names

def determine_correctness(item):
    if isinstance(item, dict):
        if "result" in item:
            if item["result"] in ["passed", "correct", "success"]:
                return 0
            else:
                return 1
        elif "pass@1" in item:
            if item["pass@1"] >= 1.0:
                return 0
            else:
                return 1
        else:
            raise Exception
    elif isinstance(item, str):
        if item in ["passed", "correct", "success"]:
            return 0
        else:
            return 1
    else:
        raise Exception
