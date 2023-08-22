import torch
from typing import List, Optional
from transformers.trainer_utils import get_last_checkpoint
import math
import os
import jsonlines
import sys
import evaluate

label_mapper = {
    'No Quote': 0,
    'Direct Quote': 1,
    'Published Work/Press Report': 2,
    'Indirect Quote': 3,
    'Statement/Public Speech': 4,
    'Background/Narrative': 5,
    'Other': 6,
    'Proposal/Order/Law': 7,
    'Email/Social Media Post': 8,
    'Court Proceeding': 9,
    'Direct Observation': 10
}

def _get_attention_mask(x: List[torch.Tensor], max_length_seq: Optional[int]=10000) -> torch.Tensor:
    max_len = max(map(lambda y: y.shape.numel(), x))
    max_len = min(max_len, max_length_seq)
    attention_masks = []
    for x_i in x:
        input_len = x_i.shape.numel()
        if input_len < max_length_seq:
            mask = torch.cat((torch.ones(input_len), torch.zeros(max_len - input_len)))
        else:
            mask = torch.ones(max_length_seq)
        attention_masks.append(mask)
    return torch.stack(attention_masks)


def load_data(args):
    here = os.path.dirname(__file__)
    file = os.path.join(here, args.dataset_name)
    train_input, val_input = [], []
    if file.endswith('.gzip'):
        outfile = file.replace('.gzip', '')
        if not os.path.exists(outfile):
            import gzip
            import shutil
            with gzip.open(file, 'rb') as f_in:
                with open(outfile, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        file = outfile

    with jsonlines.open(file) as f:
        for dat in f:
            if dat['split'] == 'train':
                train_input.append(dat)
            else:
                val_input.append(dat)

    # order, in case we want to check worst-case memory performance
    if args.dataset_order is not None:
        if args.dataset_order == 'longest-first':
            train_input = sorted(train_input, key=lambda x: -len(x))
        elif args.dataset_order == 'shortest-first':
            train_input = sorted(train_input, key=len)

    return train_input[:args.max_train_samples], val_input[:args.max_val_samples]


import numpy as np
from scipy.special import expit

def compute_metrics(eval_preds):
    metric = evaluate.load("f1")
    logits, labels = eval_preds
    if isinstance(logits, list):
        logits = np.vstack(logits)
    if isinstance(labels, list):
        labels = np.vstack(labels)

    all_preds, all_labels = [], []
    for logit_i, label_i in zip(logits, labels):
        logit_i = logit_i[logit_i != -100]
        label_i = label_i[label_i != -100]
        logit_i = logit_i.reshape(len(label_i), 11)
        pred_i = logit_i.argmax(axis=1)
        all_preds.append(pred_i)
        all_labels.append(label_i.astype(int))

    all_preds, all_labels = np.hstack(all_preds), np.hstack(all_labels)

    return metric.compute(predictions=all_preds, references=all_labels, average='macro')


def get_last_checkpoint_with_asserts(training_args):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            print(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    return last_checkpoint


def model_name_or_checkpoint(last_checkpoint, model_args):
    if last_checkpoint is not None:
        checkpoint = last_checkpoint
    elif model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path):
        checkpoint = model_args.model_name_or_path
    else:
        checkpoint = None
    return checkpoint
