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

global_eof_stops = ['<|endoftext|>', '</s>', '<｜end▁of▁sentence｜>'] # '```'

def evaluate(llm, tokenizer, dataset, generate_config, save_path, task, ext_gen_config=None):
    generate_config = copy.deepcopy(generate_config)
    if "prompt" in generate_config:
        system_prompt = generate_config.pop("prompt") + "\n    "
    else:
        system_prompt = ""
    ans_recored = shelve.open(str(save_path))

    for idx, item in enumerate(tqdm.tqdm(dataset)):
        input_str = dataset.get_prompt(idx)
        input_str = system_prompt + input_str
        #input_str += "\n    "
        generate_result = utils.generate_and_record(
            llm,
            tokenizer,
            input_str,
            generate_config=generate_config,
            extra_generation_config=ext_gen_config,
        )
        # Evaluate the code
        raw_but_no_special_token_ans = generate_result["str_output"]
        if task == "humaneval":
            post_process_string = input_str + raw_but_no_special_token_ans
        else:
            post_process_string = raw_but_no_special_token_ans
        code, min_stop_index = dataset.postprocess(post_process_string, idx)
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
    generation_config = GenerationConfig.from_pretrained(model_name,)
    generation_config.stop_strings = global_eof_stops
    if task.lower() == "humaneval":
        dataset = HumanEvalDataset()
        evaluate(model, tokenizer, dataset, generate_config, output_path, task.lower(), ext_gen_config=generation_config)
    elif task.lower() == "edit_eval":
        dataset_path = config_dict["task_config"]["dataset_path"]
        dataset = EditEvalDataset(dataset_path)
        evaluate(model, tokenizer, dataset, generate_config, output_path, task.lower(), ext_gen_config=generation_config)
    
    end = time.time()
    logger.info(f"Total time: {end - start}")

if __name__ == "__main__":
    #typer.run(main)
    app()