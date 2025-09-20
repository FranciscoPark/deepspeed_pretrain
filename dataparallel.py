import math
from functools import partial

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from datasets import Dataset
from tqdm import tqdm

from utils import *

def train(args):

    device = f'cuda' if torch.cuda.is_available() else 'cpu'

    config = AutoConfig.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_config(config)
    model = model.to(device)
    model = torch.nn.DataParallel(model)

    dataset = Dataset.from_json(args.train_dataset)
    loader = DataLoader(dataset, shuffle=True, collate_fn=clm_collate_fn, batch_size=args.batch_size)
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

    writer = SummaryWriter('logs/data_parallel')

    iterator = iter(loader)
    for step in tqdm(range(args.num_train_steps)):
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

        writer.add_scalar('Loss/train', loss.item(), step)

    writer.close()
    for rank in range(torch.cuda.device_count()):
        print(torch.cuda.memory_summary(f'cuda:{rank}'))


if __name__ == '__main__':
    args = get_args()
    train(args)
