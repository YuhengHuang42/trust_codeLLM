from loguru import logger
import typer
from typing_extensions import Annotated
from pathlib import Path
import time
import yaml
import torch
import numpy as np
import wandb
from torch.utils.data import DataLoader
import transformers
import math
import os
import torchinfo

from sae_model import Autoencoder, TopK, normalized_mean_squared_error
from extract.naive_store import NaiveTensorStore

def compute_grad_norm(model):
    total_grad_norm = 0
    for param in model.parameters():
        if param.grad is not None:  # Ensure the parameter has a gradient
            param_norm = param.grad.norm(2)  # L2 norm of the gradients
            total_grad_norm += param_norm.item() ** 2  # Sum of squares of all gradient norms

    total_grad_norm = total_grad_norm ** 0.5  # Take the square root to get the total L2 norm
    return total_grad_norm

def training_one_epoch(sae, 
                       data_loader, 
                       optimizer, 
                       loss_fn, 
                       epoch,
                       path, 
                       print_freq_prop=0.2, 
                       scheduler=None, 
                       wandb_handle=None, 
                       clip_grad=False):
    sae.train()
    loss_list = list()
    local_step_num = len(data_loader)
    cum_time = 0
    print_freq = round(local_step_num * print_freq_prop)
    for idx, store_batch in enumerate(data_loader):
        start_time = time.time()
        optimizer.zero_grad()
        batch_x = store_batch[0]
        batch_y = store_batch[1]
        # Filter out rows in x where the corresponding y is 0
        batch_x = batch_x[batch_y != 0].to(sae.device)
        latents_pre_act, latents, recons = sae(batch_x)
        total_loss = loss_fn(recons, batch_x)
        total_loss.backward()
        if clip_grad:
            pre_norm = compute_grad_norm(sae)
            #torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, sae.parameters()), 0.1)
            try:
                torch.nn.utils.clip_grad_norm_(sae.parameters(), 1, error_if_nonfinite=True)
            except:
                bug_save_path = os.path.join(path, f"error_epoch_{epoch}.pt")
                logger.error(f"Error at epoch {epoch} idx {idx}")
                logger.error("Saving to {}".format(bug_save_path))
                torch.save(
                                {'iter': epoch,
                                 "loss_list": loss_list,
                                 'model_state_dict': sae.state_dict(),
                                 'optimizer_state_dict': optimizer.state_dict(),
                                 "batch_x": batch_x.detach().to("cpu")
                                }, 
                                bug_save_path
                )
                raise Exception
            #after_norm = compute_grad_norm(sae)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
            lr = float(scheduler.get_last_lr()[0])
        else:
            lr = optimizer.param_groups[0]['lr']
        cur_loss = total_loss.detach().cpu().item()
        loss_list.append(cur_loss)
        #l2_norm = 0
        #for param in sae.parameters():
        #    l2_norm += torch.norm(param, p=2)  # L2 norm for each parameter
        if wandb_handle is not None:
                        wandb.log({
                            "epoch": epoch,
                            "Loss": loss_list[-1],
                            "LR": lr,
                            #"L2_model_params": l2_norm,
                            "grad_before_norm": pre_norm,
                            #"recons_norm": torch.norm(recons, p=2),
                            #"batch_x_norm": torch.norm(batch_x, p=2),
                        })
        end_time = time.time()
        cum_time += end_time - start_time
        if idx % print_freq == 0:
            batch_time = cum_time / idx if idx > 0 else cum_time
            cur_loss = np.average(loss_list)
            logger.info(
                f'Epoch: [{epoch}][{idx}/{local_step_num}]\t'
                f'Batch Time: {batch_time}\t'
                f'Loss: {cur_loss}'
            )
    return sae, loss_list

def evaluate(sae, data_loader, loss_fn):
    sae.eval()
    loss_list = list()
    code_list = list()
    for idx, store_batch in (enumerate(data_loader)):
        with torch.inference_mode():
            batch_x = store_batch[0]
            batch_y = store_batch[1]
            batch_x = batch_x[batch_y != 0].to(sae.device)
            #batch_x = store_batch[0].to(sae.device)

            # Forward pass
            latents_pre_act, latents, recons = sae(batch_x)
            # Get loss
            total_loss = loss_fn(recons, batch_x)
            cur_loss = total_loss.detach().cpu().item()
            loss_list.append(cur_loss)
    return np.average(loss_list)

app = typer.Typer(pretty_exceptions_show_locals=False, pretty_exceptions_short=False)

# python sae_training.py --config-file model_eval_config/CodeLlama_autoencoder.yaml --result-output-path /data/huangyuheng/trust_code/codellama34b/sae/model
@app.command()
def main(
    config_file: Annotated[Path, typer.Option()],
    result_output_path: Annotated[Path, typer.Option()],
    print_freq_prop: Annotated[float, typer.Option("--print-freq-prop")] = 0.2,
    device: Annotated[str, typer.Option("--device")] = "cuda"
):
    start = time.time()
    # ===== Load configuration =====
    with open(config_file, 'r') as file:
        config_dict = yaml.safe_load(file)
    hidden_neuron = config_dict["task_config"]["hidden_neuron"]
    storage_path = config_dict['task_config']["storage_path"]
    sae_hidden = config_dict["task_config"]["sae_hidden"]
    topk = config_dict["task_config"]["topk"]
    normalize = config_dict["task_config"]["normalize"]
    epoch_num = int(config_dict["task_config"]["epoch_num"])
    learning_rate = float(config_dict["task_config"]["learning_rate"])
    eps = config_dict["task_config"].get("eps", 6.25e-10) # https://github.com/openai/sparse_autoencoder/blob/4965b941e9eb590b00b253a2c406db1e1b193942/sparse_autoencoder/train.py#L555C46-L555C49
    batch_size = config_dict["task_config"].get("batch_size", 64)
    feature_name = config_dict["task_config"].get("feature_name", "extracted_states")
    clip_grad = config_dict["task_config"].get("clip_grad", False)
    wandb_info = config_dict["task_config"].get("wandb_info", None)
    scheduler_type = config_dict["task_config"].get("scheduler_type", None)
    tied = config_dict["task_config"].get("tied", False) 
    model_save_freq_prop = config_dict["task_config"].get("model_save_freq_prop", 1)
    model_save_freq = max(int(epoch_num * model_save_freq_prop), 1)
    
    model_setting = {
        "n_latents": sae_hidden,
        "n_inputs": hidden_neuron,
        "activation": topk,
        "normalize": normalize,
        "tied": tied
    }
    save_prefix = f"n_latents_{sae_hidden}_n_inputs_{hidden_neuron}"
    
    sae = Autoencoder(
        sae_hidden,
        hidden_neuron,
        activation=TopK(topk),
        normalize=normalize,
        tied=tied
    )
    sae = sae.to(device)
    
    
    store = NaiveTensorStore.load_from_disk(storage_path)
    
    #optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, sae.parameters()), lr=learning_rate,  eps=eps)
    optimizer = torch.optim.AdamW(sae.parameters(), lr=learning_rate,  eps=eps)
    #data_loader =  DataLoader(store, batch_size=batch_size, collate_fn=get_collate_fn(feature_name), shuffle=True)
    data_loader = store.get_data_loader(batch_size, feature_name, shuffle=True)
    if scheduler_type == "cos":
        training_step = len(data_loader) * epoch_num
        warmup_step = math.floor(training_step * 0.1) # Hard-code. 
        scheduler = transformers.get_cosine_schedule_with_warmup(optimizer=optimizer, 
                                                                num_warmup_steps=warmup_step,
                                                                num_training_steps=training_step)
    else:
        scheduler = None
    loss_fn = normalized_mean_squared_error
    if wandb_info is not None:
        wandb.login(key=wandb_info["key"])
        wandb_proj_name = wandb_info["proj_name"]
        wandb_exp_name = str(result_output_path) + f"_epoch_{epoch_num}" 
        wandbrun = wandb.init(project=wandb_proj_name, name=wandb_exp_name)
    else:
        wandbrun = None
    all_loss_list = []
    for epoch in range(epoch_num):
        sae, loss_list = training_one_epoch(sae, 
                                            data_loader, 
                                            optimizer, 
                                            loss_fn, 
                                            epoch,
                                            path=result_output_path,
                                            print_freq_prop=print_freq_prop, 
                                            scheduler=scheduler, 
                                            wandb_handle=wandbrun,
                                            clip_grad=clip_grad
                                        )
        all_loss_list += loss_list
        if epoch % model_save_freq == 0 or epoch == epoch_num - 1:
             torch.save(
                {'iter': epoch_num,
                    "loss_list": all_loss_list,
                    'model_state_dict': sae.cpu().state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                    "model_setting": model_setting
                }, 
                os.path.join(result_output_path, save_prefix + f"_epoch_{epoch}.pt")
            )
             sae = sae.to(device)
        logger.info(f'Epoch: [{epoch}] Finished. Loss: {np.average(loss_list)}\t')

    eval_loss = evaluate(sae, data_loader, loss_fn)
    if wandb_info is not None:
        wandb.log({
            "Eval loss": eval_loss,
            "model_param": torchinfo.summary(sae).trainable_params
    })
    
if __name__ == "__main__":
    typer.run(main)
    