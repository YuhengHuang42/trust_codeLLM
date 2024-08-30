import os
os.environ['HF_HOME'] = '/data/data_disk/huggingface'

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
    
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

def load_opensource_model(model_name, model_path=None, parallel=True, quantization=None):
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