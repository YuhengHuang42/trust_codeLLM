import typer
from typing_extensions import Annotated
from pathlib import Path
import yaml
import json

import task.defect4j as defects4j
import utility.utils as utils

app = typer.Typer(pretty_exceptions_show_locals=False, pretty_exceptions_short=False)

@app.command()
def main(
    important_label_path: Annotated[Path, typer.Option()],
    inference_data_path: Annotated[Path, typer.Option()],
    config_file: Annotated[Path, typer.Option()],
    task: Annotated[str, typer.Option()],
    result_output_path: Annotated[Path, typer.Option()],
    parallel: Annotated[bool, typer.Option("--parallel/--no-parallel")] = True,
):
    # ===== Load configuration =====
    with open(config_file, 'r') as file:
        config_dict = yaml.safe_load(file)
    model_name = config_dict["llm_config"]["model_name"]
    tokenizer = utils.load_tokenizer(model_name)
    
    if task.lower() == "defects4j":
        data_path = config_dict["task_config"]["repair_data_path"]
        loc_folder = config_dict["task_config"]["repair_loc_folder"]
        defects4j_path = config_dict["system_setting"]['DEFECTS4J_PATH']
        java_path = config_dict["system_setting"]['JAVA_PATH']
        dataset = defects4j.Defects4jDataset(data_path, loc_folder, defects4j_path, java_home=java_path)
        
        
        error_line_info = {}
        data = utils.load_shelve(inference_data_path)
        for key in data:
            diff_results = utils.get_changes_with_line_numbers(dataset[int(key)]['fix'], data[key]['str_output'], "java")
            error_line_info[key] =  [list(set([i[0] for i in diff_results[0]] + [i[0] for i in diff_results[1]]))]

        with open(result_output_path, 'w') as file:
            json.dump(error_line_info, file)
            
    elif task.lower() == "codetlingua":
        pass
    
