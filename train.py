from __future__ import annotations
import os
import gin
import torch
import time
import numpy as np
import torch.distributed
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import SequentialLR, LambdaLR, CosineAnnealingLR
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from huggingface_hub import hf_hub_download
from mae import *
from dataset import ImageDataset


# split paths 
TRAIN_PATH = "/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train/"
VAL_PATH = "/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/test/"
TEST_PATH = "/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/test/"


# Define the function to get image paths from a subfolder
def get_image_paths(subfolder):
    return [f"{subfolder}/{image}" for image in os.listdir(subfolder)]

# Dataset paths
train_paths = [f"{TRAIN_PATH}{path}" for path in os.listdir(f"{TRAIN_PATH}")]
test_paths = [f"{TEST_PATH}{path}" for path in os.listdir(f"{TEST_PATH}")]
val_paths = [f"{VAL_PATH}{path}" for path in os.listdir(f"{VAL_PATH}")]

# Parallelize the train set path collection
with ThreadPoolExecutor() as executor:
    # Map the function over all subfolders concurrently
    results = list(executor.map(get_image_paths, train_paths))
# Flatten the list of lists into a single list of image paths
train = list(chain.from_iterable(results))

DO_MODEL_DOWNLOAD = False  # model download flag to download the model if needed
LOAD_CHECKPOINT = False # checkpoint load flag to load the model and optimizer states if needed

# preparing the model
MODEL_REPO_ID = ""
MODEL_REPO_TYPE = ""
MODEL_FILENAME = ""

# local directory to save the downloaded files
LOCAL_DIR = "/kaggle/working"

# Download the dataset and model files if needed
if  DO_MODEL_DOWNLOAD:
    hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME, repo_type=MODEL_REPO_TYPE, local_dir=LOCAL_DIR)
    print("Model downloaded....")

if LOAD_CHECKPOINT:
    # load the checkpoint
    checkpoint = torch.load(f"{LOCAL_DIR}/{MODEL_FILENAME}", weights_only=True, map_location="cpu")
    print("Checkpoint loaded....")

def trainer(rank, world_size):
    # Enable the cudnn backend for better performance
    torch.backends.cudnn.benchmark = True

    # set the master process
    master_process = (rank == 0)
    torch.set_float32_matmul_precision('medium')  # Set the matmul precision to medium
    



    # Initialize the Process Group
    dist.init_process_group(backend=config.training_backend, rank=rank, world_size=world_size)

    # Set the Device for the Current Process
    torch.cuda.set_device(rank)
    config.model_device = torch.device("cuda", rank) # Set the device for the current process

    # prepare the training dataset
    dataset = TokenDataset(config.block_size, tokens)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True) # Use DistributedSampler to partition data among distributed processes
    dataloader = DataLoader(dataset, batch_size=config.batch_size, sampler=sampler, drop_last=True, pin_memory=True, pin_memory_device=f"{config.model_device.type}:{rank}", num_workers=1, prefetch_factor=8) # Use DataLoader to manage batches

    # Initialize the model with the configuration 
    model = GPT(config)
    if LOAD_CHECKPOINT:
        # Load the model state
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f'{rank}: Model loaded....')

    model.to(config.dtype).to(config.model_device)  # Move model to respective device and change the datatype to bfloat16
    model = DDP(model, device_ids=[rank])  # Wrap model in DDP
    
    # Define Optimizer    
    optimizer = model.module.configure_optimizers() 
    if LOAD_CHECKPOINT:
        # Load the optimizer state
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f'{rank}: Optimizer loaded....')


    # setting the total steps and warmup steps for the scheduler
    config.steps_per_epoch = len(dataloader)
    config.total_steps = (config.steps_per_epoch * config.epochs)//config.gradient_accumulation_steps
    config.warmup_steps = int(config.total_steps * config.warmup_steps_ratio)

    if master_process:
        # print loaded checkpoint if load checkpoint is True
        print(f"Total Steps: {config.total_steps}, Warmup Steps: {config.warmup_steps}")
        print(f"Steps per Epoch: {config.steps_per_epoch}, Total Tokens: {len(tokens)}")

    # Warmup scheduler
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda step: step / config.warmup_steps)

    # Cosine annealing after warmup
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=config.total_steps - config.warmup_steps, eta_min=config.eta_min)
    
    # Combine warmup and cosine
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[config.warmup_steps]) 

    try:
        # Training Loop 
        for epoch in range(config.epochs) :  # Loop over the dataset multiple times
            model.train()
            sampler.set_epoch(epoch)  # Shuffle data per epoch for distributed training 

            loss_accum, start_time = 0.0, time.time() # Initialize accumulators
            
            for batch, (inputs, labels) in enumerate(dataloader):
                gradient_accum_cond = ((batch + 1) % config.gradient_accumulation_steps == 0) or ((batch + 1) == config.steps_per_epoch)
    
                # Move data to device
                inputs, labels = inputs.to(config.model_device), labels.to(config.model_device)

                # Forward pass
                _, loss = model(inputs, labels)
                
                # scale the loss for gradient accumulation
                loss = loss / config.gradient_accumulation_steps

                # Accumulate loss
                loss_accum += loss.detach()
                
                with model.no_sync() if not gradient_accum_cond else torch.enable_grad():
                    # Backward pass
                    loss.backward() # Calculate gradients

                if gradient_accum_cond:
                    # Gradient clipping before stepping
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm_val)

                    # Update parameters
                    optimizer.step()

                    # Update learning rate for the next iteration
                    scheduler.step()
                    
                    # Zero gradients for next iteration
                    optimizer.zero_grad()

                    # all-reduce the metrics
                    dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
                    dist.all_reduce(grad_norm, op=dist.ReduceOp.AVG)

                    # time taken for the gradient accumulation step
                    end_time = time.time() - start_time

                    if master_process:
                        print(f"Epoch: {epoch}, Batch: {batch}, Loss: {loss_accum.item()}, Grad Norm: {grad_norm.item()}, Time: {end_time}")

                    # Reset accumulators
                    loss_accum, grad_norm, start_time = 0.0, None, time.time()

    except Exception as e:
        print(f"Error in training: {e}")
    try:
        # prepare the evaluation dataset  
        eval_dataset = TokenDataset(config.block_size, eval_tokens)
        eval_sampler = DistributedSampler(eval_dataset, num_replicas=world_size, rank=rank, shuffle=False) # removed drop_last=True and for dataloader also
        eval_dataloader = DataLoader(eval_dataset, batch_size=config.batch_size, sampler=eval_sampler, pin_memory=True, pin_memory_device=f"{config.model_device.type}:{rank}")

        # Validation Loop
        val_loss_accum = 0.0
        # Evaluate the model on the evaluation dataset
        with torch.no_grad():
            model.eval()
            for _, (inputs, labels) in enumerate(eval_dataloader):
                inputs, labels = inputs.to(config.model_device), labels.to(config.model_device)
                _, val_loss = model(inputs, labels)
                val_loss_accum += val_loss.detach()

        # all-reduce the metrics
        dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)

        if master_process:
            print(f"Length of Eval Dataloader: {len(eval_dataloader)}")
            print(f"Validation Loss: {val_loss_accum.item()/len(eval_dataloader)}")

    except Exception as e:
        print(f"Error in evaluation: {e}")

    # Save the model and optimizer states            
    if master_process:
        torch.save(
            {
                "name": f"{TRAIN_DATA_FILENAME}_{len(tokens)}",
                "steps": config.total_steps,
                "model_state_dict": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            "checkpoint.pth",
        )

    # Cleanup
    dist.destroy_process_group()


def run_ddp_training():
    world_size = torch.cuda.device_count()  # Number of available GPUs
    mp.spawn(trainer, args=(world_size,), nprocs=world_size, join=True)



if __name__ == "__main__":
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = str(torch.cuda.device_count())  # Total number of GPUs on this node
    print(f"{str(torch.cuda.device_count())} GPUs available")

    # Run distributed training
    run_ddp_training()