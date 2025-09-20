import argparse
import math
import torch

def get_args():
    parser = argparse.ArgumentParser()
    add_setup_args(parser)
    add_model_args(parser)
    add_train_args(parser)
    add_profile_args(parser)
    add_wandb_args(parser)
    args = parser.parse_args()
    return args

def add_wandb_args(parser):
    group = parser.add_argument_group(title='WandB')
    group.add_argument('--use_wandb', action='store_true', help='Enable logging to Weights & Biases')
    group.add_argument('--wandb_project', type=str, default='clm-training', help='wandb project name')
    group.add_argument('--wandb_run_name', type=str, default=None, help='wandb run name (optional)')
    group.add_argument('--wandb_entity', type=str, default=None, help='wandb entity (optional)')

def add_setup_args(parser):
    group = parser.add_argument_group(title='Setup')
    group.add_argument('--dataset_path', type=str, default='wikitext')
    group.add_argument('--dataset_name', type=str, default='wikitext-103-raw-v1')
    group.add_argument('--split', type=str, default='train')
    group.add_argument('--text_column', type=str, default='text')
    group.add_argument('--train_dataset', type=str, default='train_dataset.jsonl')
    group.add_argument('--train_dataset_path', type=str, default='/home/work/datasets/train-2048')
    group.add_argument('--valid_dataset_path', type=str, default='/home/work/datasets/valid-2048')
    group.add_argument("--seed", action="store", default=1203, type=int)

def add_model_args(parser):
    group = parser.add_argument_group(title='Model')
    group.add_argument('--model_name', type=str)
    group.add_argument('--tokenizer_name', type=str)
    group.add_argument('--max_seq_len', type=int, default=2048)
    group.add_argument(
        "--world_size",
        default=-1,
        type=int,
        help="number of nodes for distributed training",
    )
    group.add_argument(
        "--rank", default=-1, type=int, help="node rank for distributed training"
    )

def add_train_args(parser):
    group = parser.add_argument_group(title='Train')
    group.add_argument('--batch_size', type=int, default=528)
    group.add_argument('--lr', type=int, default=3e-5)
    group.add_argument('--num_train_epochs', type=int, default=None)
    group.add_argument('--num_train_steps', type=int, default=100)
    group.add_argument('--num_warmup_steps', type=int, default=0)
    group.add_argument('--zero_stage', type=str, default='0')
    group.add_argument("--micro_batch", action="store", default=1, type=int)
    group.add_argument("--accumulation_step", action="store", default=11, type=int)
    group.add_argument("--num_steps_per_device", action="store", default=1, type=int)
    group.add_argument('--nope_ratio', type=float, default=0.0) 
    group.add_argument("--num_valid_samples", action="store", default=50000, type=int)
    group.add_argument("--dataset_keep_in_memory", action="store_true")
    group.add_argument("--pretrain", action="store_true")
    group.add_argument("--random_nope",action="store_true")
    group.add_argument("--num_tokens", action="store", default=15600000000, type=int)
    group.add_argument("--use_custom_llama", action='store_true')
    group.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    group.add_argument('--resume', action='store_true', help='Resume training if checkpoint exists')
    group.add_argument('--save_interval', type=int, default=1, help='Save checkpoint every N epochs')
    group.add_argument('--resume_tag', type=str, default="latest", help='Checkpoint tag to resume from')
    group.add_argument("--save_by", type=str, choices=["step", "epoch"], default="epoch",
                    help="Save checkpoint by 'step' or 'epoch'")
    group.add_argument('--checkpoint_cnt', type=int, default=1, help='Save checkpoint every N epochs')
    group.add_argument("--dtype",type=str,default="bf16")
    
def add_profile_args(parser):
    group = parser.add_argument_group(title='Profile')
    group.add_argument('--profile', action='store_true')

def get_cosine_schedule_with_warmup_lr_lambda(current_step, num_warmup_steps, num_training_steps, num_cycles=0.5):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

def clm_collate_fn(batch):
    input_ids = []
    for b in batch:
        input_ids.append(torch.tensor(b["input_ids"]))
    input_ids = torch.stack(input_ids)
    return {"input_ids": input_ids}

class InfiniteIterator(object):
    def __init__(self, iterable):
        self._iterable = iterable
        self._iterator = iter(self._iterable)

    def __iter__(self):
        return self

    def __len__(self):
        return len(self._iterable)

    def __next__(self):
        next_item = None
        try:
            next_item = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self._iterable)
            next_item = next(self._iterator)
        return next_item
