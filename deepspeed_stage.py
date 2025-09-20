import os
import math
from functools import partial

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from datasets import Dataset
from tqdm import tqdm
import deepspeed
import wandb
from utils import *
import functions
from transformers.models.llama.configuration_llama import LlamaConfig
def deepspeed_stage_0(args):
    return {
        "train_micro_batch_size_per_gpu": args.micro_batch,
        "gradient_accumulation_steps": args.accumulation_step,
        "fp16": {
            "enabled": True if args.dtype=="fp16" else False,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1,
        },
        "bf16":{
            "enabled": True if args.dtype=="bf16" else False,
        },
    }


def deepspeed_stage_1(args):
    return {
        "train_micro_batch_size_per_gpu": args.micro_batch,
        "gradient_accumulation_steps": args.accumulation_step,
        "fp16": {
            "enabled": True if args.dtype=="fp16" else False,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1,
        },
        "bf16":{
            "enabled": True if args.dtype=="bf16" else False,
        },
        "zero_optimization": {
            "stage": 1,
        },
    }

def deepspeed_stage_2(args):
    return {
        "train_micro_batch_size_per_gpu": args.micro_batch,
        "gradient_accumulation_steps": args.accumulation_step,
        "fp16": {
            "enabled": True if args.dtype=="fp16" else False,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1,
        },
        "bf16":{
            "enabled": True if args.dtype=="bf16" else False,
        },
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "contiguous_gradients": True,
            "round_robin_gradients": True,
        },
    }

def deepspeed_stage_3(args):
    return {
        "train_micro_batch_size_per_gpu": args.micro_batch,
        "gradient_accumulation_steps": args.accumulation_step,
        "fp16": {
            "enabled": True if args.dtype=="fp16" else False,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1,
        },
        "bf16":{
            "enabled": True if args.dtype=="bf16" else False,
        },
        "zero_optimization": {
            "stage": 3,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True,
        },
    }

def deepspeed_stage_3_offload(args):
    return {
        "train_micro_batch_size_per_gpu": args.micro_batch,
        "gradient_accumulation_steps": args.accumulation_step,
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1,
        },
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
            "offload_param": {"device": "cpu", "pin_memory": True},
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True,
        },
    }

_dsconfig = {
    '0': deepspeed_stage_0,
    '1': deepspeed_stage_1,
    '2': deepspeed_stage_2,
    '3': deepspeed_stage_3,
    '3_off': deepspeed_stage_3_offload,
}

def compute_valid_loss(model, valid_dataloader, device, pad_token_id, config):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch in valid_dataloader:
            input_ids = batch['input_ids'].to(device)[:, :-1].contiguous()
            attention_mask = batch['attention_mask'].to(device)[:, :-1].contiguous()
            labels = batch['input_ids'][:, 1:].to(device).contiguous().view(-1)
            if pad_token_id is not None:
                labels[labels == pad_token_id] = -100
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits.contiguous().view(-1, config.vocab_size)
            loss = torch.nn.functional.cross_entropy(logits, labels, ignore_index=-100)
            total_loss += loss.item()
            num_batches += 1
    model.train()
    return total_loss / num_batches if num_batches > 0 else 0.0

def train(args):
    rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    global_rank = int(os.environ.get('RANK', 0))
    args.world_size = world_size
    # args.micro_batch = args.batch_size // world_size
    deepspeed_config = _dsconfig[args.zero_stage](args=args)
    torch.cuda.set_device(rank)
    # rank=rank, world_size=world_size
    deepspeed.init_distributed(dist_backend="nccl")
    if global_rank == 0:
        print(f"DeepSpeed Config: {deepspeed_config}")
        print(f"World Size: {world_size}")
        print(f"Micro Batch Size per GPU: {args.micro_batch}")
    # deepspeed.init_distributed(verbose=True)
    device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'

    tokenizer,model, pad_token_id,config =functions.prepare_model(args)

    train_dataset, valid_dataset = functions.prepare_dataset(args)
    train_dataloader, valid_dataloader = functions.prepare_dataloaders(args, train_dataset, valid_dataset)
    if global_rank == 0:
        print(f"Number of training samples: {len(train_dataset)}")
        print(f"Number of validation samples: {len(valid_dataset)}")
        print(f"Number of training steps per device: {args.num_steps_per_device}")
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    steps_per_epoch = args.num_steps_per_device

    if args.num_train_epochs:
        args.num_train_steps = steps_per_epoch * args.num_train_epochs
        args.num_warmup_steps = int(args.num_train_steps * 0.1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        partial(
            get_cosine_schedule_with_warmup_lr_lambda,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.num_train_steps,
        ),
        last_epoch=-1,
    )

    model, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config=deepspeed_config
    )

    step = 0
    if args.resume:
        _, client_sd = model.load_checkpoint(args.checkpoint_dir)
        if client_sd is not None and 'step' in client_sd:
            step = client_sd['step']
        if global_rank == 0:
            print(f"Resumed from step {step}")

    if args.use_wandb and global_rank == 0:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            config=vars(args),
            resume="allow" if args.resume else None
        )

    epoch = step // steps_per_epoch
    pbar = tqdm(initial=step, total=args.num_train_steps, disable=global_rank != 0)
    args.save_interval = args.num_train_steps // args.checkpoint_cnt
    while step < args.num_train_steps:
        if isinstance(train_dataloader.sampler, DistributedSampler):
            train_dataloader.sampler.set_epoch(epoch)

        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)[:, :-1].contiguous()
            attention_mask = batch['attention_mask'].to(device)[:, :-1].contiguous()
            labels = batch['input_ids'][:, 1:].to(device).contiguous().view(-1)

            if pad_token_id is not None:
                labels[labels == pad_token_id] = -100

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits.contiguous().view(-1, config.vocab_size)
            loss = torch.nn.functional.cross_entropy(logits, labels, ignore_index=-100)

            model.backward(loss)
            model.step()

            step += 1
            pbar.update(1)

            if global_rank == 0 and args.use_wandb:
                curent_lr = model.optimizer.param_groups[0]['lr']
                wandb.log({"train/loss": loss.item(), 
                            "step": step,
                            "learning_rate":curent_lr,
                        })

            # if args.save_by == "epoch":
            #     if step % steps_per_epoch == 0:
            #         current_epoch = step // steps_per_epoch
            #         if current_epoch % args.save_interval == 0 and rank == 0:
            #             print(f"[Rank {rank}] Saving checkpoint at epoch {current_epoch}")
            #             model.save_checkpoint(
            #                 args.checkpoint_dir,
            #                 tag=f"epoch{current_epoch}",
            #                 client_state={'step': step, 'epoch': current_epoch}
            #             )
            #     valid_loss = compute_valid_loss(model, valid_dataloader, device, pad_token_id, config)
            #     if rank == 0 and args.use_wandb:
            #         wandb.log({"valid/loss": valid_loss, "step": step})

            if (args.save_by == "step") and (step % args.save_interval == 0):
                print(f"[Rank {rank}] Saving checkpoint at step {step}")
                model.save_checkpoint(
                    args.checkpoint_dir,
                    tag=f"step{step}",
                    client_state={'step': step}
                )
                if (global_rank == 0):
                    valid_loss = compute_valid_loss(model, valid_dataloader, device, pad_token_id, config)
                    if args.use_wandb:
                        wandb.log({"valid/loss": valid_loss, "step": step})

            if step >= args.num_train_steps:
                break

        epoch += 1

    if global_rank == 0:
        if args.use_wandb:
            wandb.finish()
    model.save_checkpoint(args.checkpoint_dir, tag="final", client_state={'step': step})

    dist.destroy_process_group()
    # print(torch.cuda.memory_summary(f'cuda:{rank}'))

if __name__ == '__main__':
    args = get_args()
    train(args)