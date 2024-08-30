import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import json
    
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
    str_output = tokenizer.decode(seq[input_length:])
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
    