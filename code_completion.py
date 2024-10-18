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
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
app = typer.Typer(pretty_exceptions_show_locals=False, pretty_exceptions_short=False)
CODE_NOT_FOUND_FLAG = "NO_CODE"

import utility.utils as utils
from task.safim import SAFIMDataset
import task.dataset_utils as dataset_utils
# We by default use parallel for LLM loading based on all available GPUS.
# Use CUDA_VISIBLE_DEVICES=xxx to specify GPUs
def evaluate(llm, tokenizer, dataset, generate_config, save_path, model_type):
    generate_config = copy.deepcopy(generate_config)
    if "prompt" in generate_config:
        system_prompt = generate_config.pop("prompt") + "\n    "
    else:
        system_prompt = ""
    ans_recored = shelve.open(str(save_path))

    for idx, item in enumerate(tqdm.tqdm(dataset)):
        input_str = dataset.get_prompt(idx, model_type)
        input_str = system_prompt + input_str
        #input_str += "\n    "
        generate_result = utils.generate_and_record(
            llm,
            tokenizer,
            input_str,
            generate_config=generate_config
        )
        # Evaluate the code
        raw_but_no_special_token_ans = generate_result["str_output"]
        code = dataset.postprocess(raw_but_no_special_token_ans, idx)
        line_number = dataset_utils.find_buggy_positions(raw_but_no_special_token_ans, code, without_filter=True)
        code_correctness = dataset.check_result(code, idx)
        generate_result["code_correctness"] = code_correctness
        generate_result["problem"] = {"prompt": input_str} 
        generate_result["line_number"] = line_number
        generate_result["cleaned_code"] = code
        
        ans_recored[str(idx)] = generate_result
    
    ans_recored.close()

# CUDA_VISIBLE_DEVICES=0,1,2 python code_completion.py --task safim --config-file model_eval_config/CodeLlama_completion.yaml --output-path /data/huangyuheng/trust_code/codellama34b/code_completion/safim    
@app.command()
def main(
    task: Annotated[str, typer.Option()] = 'safim',
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
    
    if task.lower() == "safim":
        dataset = SAFIMDataset()
        evaluate(model, tokenizer, dataset, generate_config, output_path, model_name)
    
    end = time.time()
    logger.info(f"Total time: {end - start}")

if __name__ == "__main__":
    #typer.run(main)
    app()