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
def evaluate(llm, tokenizer, dataset, generate_config, ans_recored, iter_list=None):
    import utility.utils as utils
    generate_config = copy.deepcopy(generate_config)
    #prompt = generate_config.pop("prompt")
    if iter_list == None:
        iter_list = range(len(dataset))

    for idx in iter_list:
        prompt = dataset.get_prompt(dataset.index[idx])
        generate_result = utils.generate_and_record(
            llm,
            tokenizer,
            prompt,
            generate_config=generate_config
        )
        # Evaluate the code
        code = generate_result['str_output']
        #if code is None:
        #    code_correctness = CODE_NOT_FOUND_FLAG
        #else:
        #    code_correctness = dataset.check_result(code, idx)
        code_correctness = dataset.check_result(code, idx)
        generate_result["code_correctness"] = code_correctness
        generate_result["problem"] = {"code": dataset[idx]['buggy'], "prompt": prompt, "gt": dataset[idx]["fix"]} 
        
        ans_recored[str(idx)] = generate_result
    
#  python3 code_repair.py --config-file model_eval_config/CodeLlama_repair.yaml --output-path /data/huangyuheng/trust_code/codellama34b/repair/repair
@app.command()
def main(
    task: Annotated[str, typer.Option()] = 'defects4j',
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
    import utility.utils as utils
    from task.defect4j import Defects4jDataset
    data_path = config_dict["task_config"]["repair_data_path"]
    loc_folder = config_dict["task_config"]["repair_loc_folder"]
    defects4j_path = config_dict["system_setting"]['DEFECTS4J_PATH']
    java_path = config_dict["system_setting"]['JAVA_PATH']

    
    ## Load model
    model_name = config_dict["llm_config"]["model_name"]
    quantization = config_dict["llm_config"]["quantization"]
    generate_config = config_dict["llm_config"]["generate_config"]
    model, tokenizer = utils.load_opensource_model(model_name, parallel=parallel, quantization=quantization)
    
    if task == "defects4j":
        assert "split" in config_dict["task_config"]
        dataset = Defects4jDataset(data_path, loc_folder, defects4j_path, java_home=java_path)
        already_saved_length, saved_path = utils.load_shelve_and_resume(os.path.dirname(str(output_path)))
        if already_saved_length == 0:
            ans_recored = shelve.open(str(output_path))
            iter_list = None
        else:
            logger.warning(f"Saved data already exists in {os.path.dirname(str(output_path))}. Resume from it. Start from {already_saved_length}.")
            logger.warning(f"Please make sure this is expected. We anticipate there is only one file under a folder")
            ans_recored = shelve.open(str(saved_path))
            iter_list = range(already_saved_length, len(dataset))
        evaluate(model, tokenizer, dataset, generate_config, ans_recored, iter_list)
        ans_recored.close()
    
    end = time.time()
    logger.info(f"Total time: {end-start}")

if __name__ == "__main__":
    #typer.run(main)
    app()