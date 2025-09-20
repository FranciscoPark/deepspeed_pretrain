import os
import math
from functools import partial

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from datasets import Dataset
from tqdm import tqdm

from utils import *

def train(args):

    rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)

    device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'

    config = AutoConfig.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_config(config)
    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    dataset = Dataset.from_json(args.train_dataset)
    sampler = DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(dataset, sampler=sampler, collate_fn=clm_collate_fn, batch_size=args.batch_size // world_size)
    loader = InfiniteIterator(loader)

    if args.num_train_epochs:
        args.num_train_steps = len(loader) * args.num_train_epochs

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

    writer = SummaryWriter('logs/distributed_data_parallel') if rank == 0 else None

    iterator = iter(loader)
    for step in tqdm(range(args.num_train_steps), disable=rank != 0):
        batch = next(loader)
        batch = batch['input_ids'].to(device)
        input_ids, target = batch[:,:-1], batch[:,1:]
        output = model(input_ids)

        logit = output.logits
        logit = logit.contiguous().view(-1, logit.shape[-1])
        target = target.contiguous().view(-1)

        loss = torch.nn.functional.cross_entropy(logit, target)

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        step += 1

        if rank == 0: writer.add_scalar('Loss/train', loss.item(), step)

    if rank == 0: writer.close()
    dist.destroy_process_group()

    print(torch.cuda.memory_summary(device))

if __name__ == '__main__':
    args = get_args()
    train(args)
