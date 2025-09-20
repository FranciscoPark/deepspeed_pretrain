import os
from itertools import chain

from transformers import AutoTokenizer
from datasets import load_dataset
from datasets import Dataset, Features, Sequence, Value

from utils import get_args

def tokenize_function(examples, tokenizer, text_column):
    if text_column not in examples:
        print(f"Missing text column: {text_column}")
        print(f"Available keys: {examples.keys()}")
    output = tokenizer(examples[text_column])
    return output

def group_texts(examples, block_size):
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    result = {
	k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
	for k, t in concatenated_examples.items()
    }
    return result

def main(args):

    num_proc = os.cpu_count()

    dataset = load_dataset(
                args.dataset_path,
                name=args.dataset_name,
                split=args.split
            )
    column_names = list(dataset.features)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    
    print(len(dataset))
    dataset = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=column_names,
                num_proc=num_proc,
                fn_kwargs={
                    'tokenizer': tokenizer,
                    'text_column': args.text_column,
                },
                desc='Tokenize Texts'
            )
    print(dataset[0])
    print("Final dataset length before second map:", len(dataset))
    dataset = dataset.map(
                group_texts,
                batched=True,
                num_proc=num_proc,
                fn_kwargs={
                    'block_size': tokenizer.model_max_length,
                },
                desc='Group Texts'
            )
    print("Final dataset length before writing:", len(dataset))
    
    features = Features({
        "input_ids": Sequence(Value("int32")),
        "attention_mask": Sequence(Value("int8")),
    })
    dataset = Dataset.from_dict(dataset[:], features=features)
    dataset.to_json(args.train_dataset)
    

if __name__ == '__main__':
    args = get_args()
    main(args)
