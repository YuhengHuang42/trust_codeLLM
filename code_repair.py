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

from utility.utils import HARD_TOKEN_LIMIT
import utility.utils as utils
from task.defect4j import Defects4jDataset
from task.quixbug import QuixbugDatasetPy
from task.evalpack_repair import HumanEvalPackRepair

app = typer.Typer(pretty_exceptions_show_locals=False, pretty_exceptions_short=False)

from utility.utils import CODE_NOT_FOUND_FLAG

global_eof_stops = ['// Buggy Function', '// Fixed Function', '# Buggy Function', '# Fixed Function',
                    '/* Buggy Function */', '/* Fixed Function */', '<|endoftext|>']

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def evaluate(llm, 
             tokenizer, 
             dataset, 
             generate_config, 
             ans_recored, 
             iter_list=None, 
             ext_gen_config=None,
             ):
    import utility.utils as utils
    generate_config = copy.deepcopy(generate_config)
    #prompt = generate_config.pop("prompt")
    if iter_list == None:
        iter_list = range(len(dataset))
    
    if ext_gen_config is not None:
        stop_strings = ext_gen_config.stop_strings
    else:
        stop_strings = None
    for idx in tqdm.tqdm(iter_list):
        prompt, ref_func = dataset.get_prompt(idx)
        max_new_tokens = int(2*len(tokenizer.encode(ref_func, return_tensors='pt', add_special_tokens=False)[0])) # reference:
        max_token_all = len(tokenizer.encode(prompt, return_tensors='pt')[0]) + max_new_tokens
        if max_token_all > HARD_TOKEN_LIMIT:
            continue
        generate_config["max_new_tokens"] = max_new_tokens
        generate_result = utils.generate_and_record(
            llm,
            tokenizer,
            prompt,
            generate_config=generate_config,
            extra_generation_config=ext_gen_config,
            remove_seq_list=stop_strings
        )
        # Evaluate the code
        code = generate_result['str_output']
        #if code is None:
        #    code_correctness = CODE_NOT_FOUND_FLAG
        #else:
        #    code_correctness = dataset.check_result(code, idx)
        code_correctness = dataset.check_result(code, idx)
        generate_result["code_correctness"] = code_correctness
        generate_result["problem"] = {"code": dataset.get_buggy_code(idx), 
                                      "prompt": prompt, 
                                      "gt": dataset.get_fix_code(idx)
                                      } 
        
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
    if "cache_dir" in config_dict["system_setting"]:
        cache_dir = config_dict["system_setting"]["cache_dir"]
    else:
        cache_dir = None
    data_path = config_dict["task_config"].get("repair_data_path", None)
    loc_folder = config_dict["task_config"].get("repair_loc_folder", None)
    defects4j_path = config_dict["system_setting"].get('DEFECTS4J_PATH', None)
    java_path = config_dict["system_setting"].get('JAVA_PATH', None)

    
    ## Load model
    model_name = config_dict["llm_config"]["model_name"]
    quantization = config_dict["llm_config"]["quantization"]
    generate_config = config_dict["llm_config"]["generate_config"]
    model, tokenizer = utils.load_opensource_model(model_name, parallel=parallel, quantization=quantization, cache_dir=cache_dir)
    
    if task.lower() == "defects4j":
        assert "split" in config_dict["task_config"]
        dataset = Defects4jDataset(data_path, loc_folder, defects4j_path, java_home=java_path)
        already_saved_length, saved_path = utils.load_shelve_and_resume(os.path.dirname(str(output_path)))
        generation_config = GenerationConfig.from_pretrained(model_name,)
        generation_config.stop_strings = global_eof_stops + ["// Provide a fix for the buggy function"]
    elif task.lower() == "quixbug":
        dataset = QuixbugDatasetPy(data_path, "/tmp/quixbug", loc_folder)
        already_saved_length, saved_path = utils.load_shelve_and_resume(os.path.dirname(str(output_path)))
        generation_config = GenerationConfig.from_pretrained(model_name,)
        generation_config.stop_strings = global_eof_stops + ["# Provide a fix for the buggy function"]
    elif "evalpack" in task.lower():
        dataset = HumanEvalPackRepair()
        already_saved_length, saved_path = utils.load_shelve_and_resume(os.path.dirname(str(output_path)))
        generation_config = GenerationConfig.from_pretrained(model_name,)
        generation_config.stop_strings = ["def check"]
    if already_saved_length == 0:
        ans_recored = shelve.open(str(output_path))
        iter_list = None
    else:
        logger.warning(f"Saved data already exists in {os.path.dirname(str(output_path))}. Resume from it. Start from {already_saved_length}.")
        logger.warning(f"Please make sure this is expected. We anticipate there is only one file under a folder")
        ans_recored = shelve.open(str(saved_path))
        iter_list = range(already_saved_length, len(dataset))
    evaluate(model, tokenizer, dataset, generate_config, ans_recored, iter_list, ext_gen_config=generation_config)
    ans_recored.close()
    
    end = time.time()
    logger.info(f"Total time: {end-start}")

if __name__ == "__main__":
    #typer.run(main)
    app()