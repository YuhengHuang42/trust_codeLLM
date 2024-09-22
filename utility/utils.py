import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from collections import OrderedDict
import torch
import json
from pygments.lexers import get_lexer_by_name
from pygments.token import Token
import numpy as np
  
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
    
    start_idx = None
    end_idx = None
    for idx, (left, right) in enumerate(original_token_info['offset_mapping']):
        if start_idx is None and (start_pos >= left and start_pos < right):
            start_idx = idx
        if end_idx is None and (end_pos >= left and end_pos < right):
            end_idx = idx
            break
    assert start_idx is not None
    assert end_idx is not None
    
    return (start_idx, end_idx)

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

def generate_and_record(llm, tokenizer, input_str, generate_config={"max_new_tokens": 600}):
    # https://huggingface.co/docs/transformers/v4.44.2/en/main_classes/text_generation#transformers.GenerationMixin.generate
    # https://huggingface.co/docs/transformers/v4.44.2/en/main_classes/text_generation#transformers.GenerationConfig
    
    inputs = tokenizer(input_str, return_tensors="pt").to(llm.device)
    input_length = inputs["input_ids"].shape[-1]
    result = {"input_length": input_length}
    if "return_dict_in_generate" in generate_config:
        generate_config["output_logits"] = True
    with torch.inference_mode():
        output = llm.generate(
            inputs["input_ids"],
            pad_token_id=tokenizer.eos_token_id,
            **generate_config
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
    result["str_all"] = str_all
    result["str_output"] = str_output
    token_output = seq.tolist()
    result["token_output"] = token_output
    return result



def load_tokenizer(model_name, model_path=None):
    if model_name == "Phind/Phind-CodeLlama-34B-v2":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

def load_opensource_model(model_name, parallel=True, quantization=None, model_path=None):
    if parallel:
        args = {"device_map": "auto"}
    else:
        args = {}
    
    assert quantization in [None, "8bit", "4bit"]
    if quantization == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
            #load_in_8bit=True,
            #bnb_8bit_compute_dtype=torch.float16
        )
    elif quantization == "8bit":
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        quantization_config = None
    
    tokenizer = load_tokenizer(model_name)
    if model_name == "Phind/Phind-CodeLlama-34B-v2":
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
    
def generate_jsonl_for_openai(request_id_list, 
                              message_list, 
                              output_path,
                              max_tokens=None, 
                              model_type="gpt-4o-mini", 
                              url="/v1/chat/completions"):
    """
    Prepare input batch data for OPENAI API
    Args:
        request_id_list: used to index the message_list.
        message_list: list of messages to be sent to the API
        output_path: output file path
        max_tokens: maximum tokens to generate
        model_type: model type of OPENAI API
        url: API endpoint
    """
    assert len(request_id_list) == len(message_list)
    data = []
    requestid_to_message = dict(zip(request_id_list, message_list))
    for idx, item in enumerate(request_id_list):
        request_id = f"request-{item}"
        message = message_list[idx]
        body = {"model": model_type, 
         "messages": [{"role": "system", "content": message["system"]},
                      {"role": "user", "content": message["user"]}],
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        per_request = {
            "custom_id": request_id,
            "method": "POST",
            "url": url,
            "body": body
        }
        data.append(per_request)
    
    write_jsonl(output_path, data)
    
    return data, requestid_to_message

def submit_batch_request_openai(client, 
                                input_file_path, 
                                url="/v1/chat/completions", 
                                completion_window="24h", 
                                description="code analysis"):
    """
    Submit the batch task to OPENAI API
    Args:
        client: OPENAI API client (client = openai.OpenAI(api_key=api_key))
        input_file_path: input file path
        url: API endpoint
        completion_window: completion window
        description: description of the submitted task
    """
    batch_input_file = client.files.create(
        file=open(input_file_path, "rb"),
        purpose="batch"
    )
    batch_input_file_id = batch_input_file.id
    batch_submit_info = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint=url,
        completion_window=completion_window,
        metadata={
        "description": description
        }
    )
    batch_submit_info_id = batch_submit_info.id
    batch_result_info = client.batches.retrieve(batch_submit_info_id)
    
    return (batch_submit_info, batch_result_info)

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
        important_line_info: dict. data_index -> line mapping
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
        tokenized_info = tokenizer(data[key]['str_output'], return_offsets_mapping=True, add_special_tokens=False)
        for item in single_info:
            start_line_number = item[0]
            end_line_number = item[1]
            cleaned_line_number = list()
            for inner_line_idx in range(start_line_number, end_line_number):
                if is_line_only_punctuators_pygments(split_response[inner_line_idx]):
                    continue
                else:
                    cleaned_line_number.append(inner_line_idx)
            
            mapping_result = [line_char_mapping[i] for i in cleaned_line_number]
            target_buggy_positions[key] += mapping_result
            important_token_info[key].append([])
            for single in mapping_result:
                important_token_info[key][-1] += get_token_indices(tokenized_info["offset_mapping"], single[0], single[1])
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
    mean_result = [agg_method(i) for i in result]
    rank_per_line = sorted(list(zip(mean_result, [i for i in range(len(mean_result))])))[:topk]
    selected_token = set()
    for line in rank_per_line:
        selected_token = selected_token.union(set(candidate_tokens[line[1]]))
    return selected_token

def load_shelve(path):
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
    import os
    import shelve
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