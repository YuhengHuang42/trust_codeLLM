from loguru import logger
import typer
from typing_extensions import Annotated
from pathlib import Path
import time
import yaml
import torch
import numpy as np
import tqdm

from detect_model import PCAEncoder
from extract.naive_store import NaiveTensorStore, VariedKeyTensorStore


def training_encoder(encoder: PCAEncoder, data_loader_list):
    for data_loader in data_loader_list:
        for idx, store_batch in tqdm.tqdm(enumerate(data_loader)):
            original, mutated, original_index, mutated_index, additional_info = store_batch
            mutated_index = mutated_index.int()
            input_data = [original, mutated[mutated_index]]
            input_data = torch.cat(input_data)
            encoder.partial_fit(input_data)
            
    return encoder
            
app = typer.Typer(pretty_exceptions_show_locals=False, pretty_exceptions_short=False)

# python sae_training.py --config-file model_eval_config/CodeLlama_autoencoder.yaml --result-output-path /data/huangyuheng/trust_code/codellama34b/sae/contrastive_model
# python sae_training.py --config-file model_eval_config/CodeLlama_autoencoder.yaml --result-output-path /data/huangyuheng/trust_code/codellama34b/sae/contrastive_model_tracer
@app.command()
def main(
    config_file: Annotated[Path, typer.Option()],
    save_path: Annotated[Path, typer.Option()],
):
    start = time.time()
    # ===== Load configuration =====
    with open(config_file, 'r') as file:
        config_dict = yaml.safe_load(file)
    target_dim = config_dict["task_config"]["target_dim"]
    storage_paths = config_dict['task_config']["storage_paths"]
    additional_storage_paths = config_dict['task_config'].get("additional_storage_paths", None)
    batch_size = config_dict["task_config"].get("batch_size", 64)
    feature_name = config_dict["task_config"].get("feature_name", "extracted_states")
    store_type = config_dict["task_config"].get("store_type", "naive")
    num_workers = config_dict["task_config"].get("num_workers", 4)
    prefetch_factor = config_dict["task_config"].get("prefetch_factor", 2)
    assert store_type in ["naive", "varied"]

    


    encoder = PCAEncoder()
    encoder.prepare_for_train(target_dim=target_dim)
    
    store_list = []
    if store_type == "naive":
        for path in storage_paths:
            store_list.append(NaiveTensorStore.load_from_disk(path))
        #store = NaiveTensorStore.load_from_disk(storage_path)
    else:
        for path in storage_paths:
            store_list.append(VariedKeyTensorStore.load_from_disk(path, open_mode="r"))
        #store = VariedKeyTensorStore.load_from_disk(storage_path, open_mode="r")
    
    additional_storage_list = []
    if additional_storage_paths is not None:
        if store_type == "naive":
            for path in additional_storage_paths:
                additional_storage_list.append(NaiveTensorStore.load_from_disk(path))
            #store = NaiveTensorStore.load_from_disk(storage_path)
        else:
            for path in additional_storage_paths:
                additional_storage_list.append(VariedKeyTensorStore.load_from_disk(path, open_mode="r"))
    
    logger.info("Storage Ready")
    #data_loader =  DataLoader(store, batch_size=batch_size, collate_fn=get_collate_fn(feature_name), shuffle=True)
    data_loader_list = []
    for store in store_list:
        data_loader_list.append(store.get_data_loader(batch_size, feature_name, shuffle=True, num_workers=num_workers, prefetch_factor=prefetch_factor))
    additional_loader_list = []
    for store in additional_storage_list:
        additional_loader_list.append(store.get_data_loader(batch_size, feature_name, shuffle=True, num_workers=num_workers, prefetch_factor=prefetch_factor))
  

    encoder = training_encoder(encoder, data_loader_list)
        
    end = time.time()
    logger.info(f"Total time: {end - start}")
    
    torch.save(encoder.state_dict(), save_path)
    
if __name__ == "__main__":
    typer.run(main)
    