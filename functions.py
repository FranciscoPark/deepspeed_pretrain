import argparse
import os
import gc
import psutil
from easydict import EasyDict
import sys
import json
import math
import wandb
import itertools
import tempfile
from tqdm import tqdm
import os
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from custom_llama import LlamaForCausalLM
os.environ["WANDB__SERVICE_WAIT"] = "300"
import datasets
import numpy as np
import random

import torch
import torch.distributed as dist

class MyLlamaConfig(LlamaConfig):
    def __init__(self, nope_ratio=1.0, **kwargs):
        super().__init__(**kwargs)
        self.nope_ratio = nope_ratio

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    

def prepare_dataset(args):
    set_seed(args.seed)
    def get_dataset_indices(args):
       
        num_docs = math.ceil(args.num_tokens / args.max_seq_len)
        indices = range(0, num_docs)
        args.num_docs = num_docs
            
        return indices    
    indices = get_dataset_indices(args)
    train_dataset = datasets.load_from_disk(args.train_dataset_path)
    train_dataset = train_dataset.shuffle(seed=args.seed)
    train_dataset = train_dataset.select(
        indices=indices,
        keep_in_memory=True if args.dataset_keep_in_memory else False
    )

    valid_dataset = datasets.load_from_disk(args.valid_dataset_path)
    valid_dataset = valid_dataset.shuffle(seed=args.seed)
    valid_dataset = valid_dataset.select(
        [i for i in range(args.num_valid_samples)],
        keep_in_memory=True
    )
    return train_dataset, valid_dataset

def prepare_dataloaders(args, train_dataset, valid_dataset):
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=args.world_size)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset, num_replicas=args.world_size)
    
    def clm_collate_fn(batch):
        input_ids = []
        attn_masks = []
        for b in batch:
            input_ids.append(torch.tensor(b["input_ids"]))
            attn_masks.append(torch.tensor(b["attention_mask"]))
        input_ids = torch.stack(input_ids)
        attn_masks = torch.stack(attn_masks)
        return {"input_ids": input_ids,
                "attention_mask":attn_masks,
                # "labels":input_ids
                }
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.micro_batch,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
        collate_fn=clm_collate_fn,
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.micro_batch,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=valid_sampler,
        drop_last=True,
        collate_fn=clm_collate_fn,
    )
    args.num_steps_per_device = math.ceil(len(train_dataloader)/args.accumulation_step) 
     
    return train_dataloader, valid_dataloader



def prepare_model(args):
    set_seed(args.seed)
    # MyLlamaConfig.register_for_auto_class()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    # config_class=MyLlamaConfig
    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    config.nope_ratio = args.nope_ratio  
    config.enable_head_metrics=False
    if args.random_nope:
        config.random_nope =True

    model_class = LlamaForCausalLM if args.use_custom_llama else AutoModelForCausalLM
    model = (model_class.from_config(config) if args.pretrain 
             else model_class.from_pretrained(args.model_name, trust_remote_code=True,config=config))
    
    return tokenizer, model, pad_token_id, config