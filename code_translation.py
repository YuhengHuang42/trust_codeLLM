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

app = typer.Typer(pretty_exceptions_show_locals=False, pretty_exceptions_short=False)
CODE_NOT_FOUND_FLAG = "NO_CODE"

# We by default use parallel for LLM loading based on all available GPUS.
# Use CUDA_VISIBLE_DEVICES=xxx to specify GPUs
def evaluate(llm, tokenizer, dataset, generate_config, save_path):
    generate_config = copy.deepcopy(generate_config)
    prompt = generate_config.pop("prompt")
    ans_recored = shelve.open(str(save_path))

    for idx, item in enumerate(tqdm.tqdm(dataset)):
        problem = item["code"]
        input_str = prompt + problem
        generate_result = utils.generate_and_record(
            llm,
            tokenizer,
            input_str,
            generate_config=generate_config
        )
        # Evaluate the code
        code = dataset_utils.extract_code_block(generate_result['str_output'])
        if code is None:
            code_correctness = CODE_NOT_FOUND_FLAG
        else:
            code_correctness = dataset.check_result(code, idx)
        generate_result["code_correctness"] = code_correctness
        generate_result["problem"] = {"code": problem, "prompt": prompt} 
        
        ans_recored[str(idx)] = generate_result
    
    ans_recored.close()
    
@app.command()
def main(
    task: Annotated[str, typer.Option()] = 'codetlingua',
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
    import utils
    import dataset_utils
    
    ## Load model
    model_name = config_dict["llm_config"]["model_name"]
    quantization = config_dict["llm_config"]["quantization"]
    generate_config = config_dict["llm_config"]["generate_config"]
    model, tokenizer = utils.load_opensource_model(model_name, parallel=parallel, quantization=quantization)
    
    if task == "codetlingua":
        assert "split" in config_dict["task_config"]
        dataset = dataset_utils.CodetlinguaDataset(split=config_dict["task_config"]["split"], 
                                                   source_lang=config_dict["task_config"]["source_lang"], 
                                                   target_lang=config_dict["task_config"]["target_lang"],
                                                   )
        evaluate(model, tokenizer, dataset, generate_config, output_path)

if __name__ == "__main__":
    #typer.run(main)
    app()