from pathlib import Path
from typing_extensions import Annotated
import typer
from loguru import logger
import time
import yaml
import tqdm
import copy
import shelve
import os
from transformers import GenerationConfig
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
app = typer.Typer(pretty_exceptions_show_locals=False, pretty_exceptions_short=False)

import utility.utils as utils
from utility.utils import CODE_NOT_FOUND_FLAG
from task.human_eval import HumanEvalDataset
from task.edit_eval import EditEvalDataset
# We by default use parallel for LLM loading based on all available GPUS.
# Use CUDA_VISIBLE_DEVICES=xxx to specify GPUs

global_eof_stops = [
    "<|endoftext|>",
    "<|endofmask|>",
    "</s>",
    "\nif __name__",
    "\ndef main(",
    "\nprint(",
]

# Ref: https://github.com/evalplus/evalplus/blob/f3bb2b13093558dd7528f769e37c0b460e319bfa/evalplus/provider/utility.py#L26
def extra_eos_for_direct_completion(dataset):
    if dataset.lower() == "humaneval":
        return ["\ndef ", "\nclass ", "\nimport ", "\nfrom ", "\nassert "]
    elif dataset.lower() == "mbpp":
        return ['\n"""', "\nassert"]
    raise ValueError(f"Unknown dataset: {dataset}")

def evaluate(llm, tokenizer, dataset, generate_config, save_path, task, ext_gen_config=None, extract_code=False):
    generate_config = copy.deepcopy(generate_config)
    prompt_template = None
    if "prompt" in generate_config:
        system_prompt = generate_config.pop("prompt") + "\n"
    elif "prompt_template" in generate_config:
        prompt_template_info = generate_config.pop("prompt_template")
        prompt_template = prompt_template_info["template"]
        instruction = prompt_template_info["instruction"]
        response_prefix = prompt_template_info["response_prefix"]
    else:
        system_prompt = ""
    ans_recored = shelve.open(str(save_path))
    
    if ext_gen_config is not None:
        stop_strings = ext_gen_config.stop_strings
    else:
        stop_strings = None
        
    for idx, item in enumerate(tqdm.tqdm(dataset)):
        input_str = dataset.get_prompt(idx)
        if prompt_template is None:
            input_str = system_prompt + input_str
        else:
            input_str = prompt_template.format(instruction, input_str, response_prefix)
        #input_str += "\n    "
        generate_result = utils.generate_and_record(
            llm,
            tokenizer,
            input_str,
            generate_config=generate_config,
            extra_generation_config=ext_gen_config,
            remove_seq_list=stop_strings
        )
        # Evaluate the code
        raw_but_no_special_token_ans = generate_result["str_output"]
        if task == "humaneval":
            post_process_string = input_str + raw_but_no_special_token_ans
        else:
            post_process_string = raw_but_no_special_token_ans
        if extract_code:
            code_blocks, code_blocks_info = utils.extract_code_block(raw_but_no_special_token_ans)
            if code_blocks is None or len(code_blocks) == 0:
                code_correctness = CODE_NOT_FOUND_FLAG
            else:
                code = code_blocks[0]
                min_stop_index = code_blocks_info[0][-1]
        else:
            code, min_stop_index = dataset.postprocess(post_process_string, idx, real_prompt=input_str)
        code_correctness = dataset.check_result(code, idx)
        generate_result["code_correctness"] = code_correctness
        generate_result["problem"] = {"prompt": input_str} 
        generate_result["stop_index"] = min_stop_index
        generate_result["cleaned_code"] = code
        
        ans_recored[str(idx)] = generate_result
    
    ans_recored.close()

# CUDA_VISIBLE_DEVICES=0,1,2 python code_generation.py --task humaneval --config-file model_eval_config/CodeLlama_generation.yaml --output-path /data/huangyuheng/trust_code/codellama34b/code_gen/humaneval    
@app.command()
def main(
    task: Annotated[str, typer.Option()] = 'humaneval',
    config_file: Annotated[Path, typer.Option()] = None,
    output_path: Annotated[Path, typer.Option()] = None,
    parallel: Annotated[bool, typer.Option("--parallel/--no-parallel")] = True,
):
    
    logger.info(f"Evaluate {task}")
    start = time.time()
    with open(config_file, 'r') as file:
        config_dict = yaml.safe_load(file)

    if "HF_HOME" in config_dict["system_setting"]:
        os.environ["HF_HOME"] = config_dict["system_setting"]["HF_HOME"]
    if "cache_dir" in config_dict["system_setting"]:
        cache_dir = config_dict["system_setting"]["cache_dir"]
    else:
        cache_dir = None
    
    ## Load model
    model_name = config_dict["llm_config"]["model_name"]
    quantization = config_dict["llm_config"]["quantization"]
    generate_config = config_dict["llm_config"]["generate_config"]
    model, tokenizer = utils.load_opensource_model(model_name, parallel=parallel, quantization=quantization, cache_dir=cache_dir)
    extract_code = config_dict["llm_config"].get("extract_code", False)
    if extract_code is True:
        logger.info("Extract code from the generated text")
    generation_config = GenerationConfig.from_pretrained(model_name,)
    generation_config.stop_strings = global_eof_stops
    if task.lower() == "humaneval" and generate_config.get("prompt_template", None) is None:
        generation_config.stop_strings += extra_eos_for_direct_completion(task.lower())
    if task.lower() == "humaneval":
        dataset = HumanEvalDataset()
        evaluate(model, tokenizer, dataset, generate_config, output_path, task.lower(), ext_gen_config=generation_config, extract_code=extract_code)
    elif task.lower() == "edit_eval":
        dataset_path = config_dict["task_config"]["dataset_path"]
        dataset = EditEvalDataset(dataset_path)
        evaluate(model, tokenizer, dataset, generate_config, output_path, task.lower(), ext_gen_config=generation_config, extract_code=extract_code)
    
    end = time.time()
    logger.info(f"Total time: {end - start}")

if __name__ == "__main__":
    #typer.run(main)
    app()