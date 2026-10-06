from tqdm.auto import tqdm
from run_clm import get_datasets, preprocess_datasets, TrainingArguments
from arguments import DataTrainingArguments
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import numpy as np
import argparse
import torch, math


def get_data(data_args, training_args, tokenizer, config):
    ## load dataset
    data_files = {}
    data_files["train"] = data_args.train_file
    extension = (
        data_args.train_file.split(".")[-1]
        if data_args.train_file is not None
        else data_args.validation_file.split(".")[-1]
    )
    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
    )
    raw_datasets["validation"] = load_dataset(
        extension,
        data_files=data_files,
        split=f"train[:{data_args.validation_split_percentage}%]",
    )
    raw_datasets["train"] = load_dataset(
        extension,
        data_files=data_files,
        split=f"train[{data_args.validation_split_percentage}%:]",
    )
    train_dataset, eval_dataset = preprocess_datasets(raw_datasets, tokenizer, data_args, training_args, config)
    return raw_datasets, train_dataset, eval_dataset


if __name__ == '__main__':
    # set up args
    output_files = [
        'test-affiliation-batched-labels-special-prefix'
        'test-argumentation-batched-labels-special',
        'test-discourse-batched-labels-special-prefix',
        'test-kmeans-clustered-batched-labels-special-prefix',
        'test-nli-batched-labels-special-prefix',
        'test-random-batched-labels-special-prefix',
        'test-role-batched-labels-special-prefix',
        'test-source-type-batched-labels-special-prefix',
        'test-stance-batched-labels-special-prefix',
    ]
    #
    data = [
        'affiliation-formatted-data.csv',
        'argumentation-formatted-data.csv',
        'clustered-formatted-data.csv',
        'discourse-formatted-data.csv',
        'nli-formatted-data.csv',
        'random-formatted-data.csv',
        'role-formatted-data.csv',
        'source-type-formatted-data.csv',
        'stance-formatted-data.csv',
    ]

    p = argparse.ArgumentParser()
    p.add_argument('--output-dir', type=str, default='tmp')
    p.add_argument('--batch-lines', action='store_true')
    args = p.parse_args()

    for model_output_file, data_file in zip(output_files, data):
        data_args = DataTrainingArguments(
            train_file=data_file,
            preprocessing_num_workers=1,
            group_texts_by='concatenate' if args.batch_lines else None,
            no_loss_on_prefix=True
        )
        training_args = TrainingArguments(output_dir='tmp', do_train=True, do_eval=True)

        # load models
        config = AutoConfig.from_pretrained(model_output_file)
        model = AutoModelForCausalLM.from_pretrained(model_output_file)
        tokenizer = AutoTokenizer.from_pretrained(model_output_file)
        eval_dataset = get_datasets(tokenizer, data_args, evaluate=True)

        model = model.to('cuda')
        ppls = []
        for datum in tqdm(eval_dataset):
            datum = {k: torch.tensor(v).to('cuda') for k, v in datum.items()}
            output = model(**datum)
            ppl = math.exp(output.loss)
            ppls.append(ppl)
