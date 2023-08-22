import os
import json, jsonlines

from transformers import (
    AutoTokenizer, AutoConfig, AutoModel, Trainer, TrainingArguments, HfArgumentParser,
)
import sys
sys.path.insert(0, '.')
from arguments import RunnerArguments, ModelArguments, DatasetArguments
from sentence_util import (
    compute_metrics,
    load_data,
    get_last_checkpoint_with_asserts,
    model_name_or_checkpoint,
    label_mapper
)


if __name__ == '__main__':
    parser = HfArgumentParser((RunnerArguments, ModelArguments, DatasetArguments, TrainingArguments,))
    runner_args, model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    from sentence_model import SentenceClassificationModel as ModelClass
    from sentence_model import TokenizedDataset, collate_fn

    # weird bug
    if training_args.report_to == ['null']:
        training_args.report_to = []

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    hf_model = (
        AutoModel
            .from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
            .to(device=training_args.device)
    )

    config.context_layer = model_args.context_layer
    if config.context_layer == 'transformer':
        context_config = AutoConfig.from_pretrained(model_args.model_name_or_path)
        context_config.num_attention_heads = 2
        context_config.num_hidden_layers = 2
        context_config.max_position_embeddings = 120
        config.context_config = context_config.to_dict()

    config.frozen_layers = model_args.freeze_layers
    config.classification_head = {
        'num_labels': max(label_mapper.values()) + 1,
        'pooling_method': model_args.pooling_method,
    }

    model = ModelClass(config=config, hf_model=hf_model)

    # Load the data
    train_input, val_input = load_data(data_args)
    train_dataset = TokenizedDataset(
        train_input, tokenizer, max_length=data_args.max_sequence_len, label_mapper=label_mapper
    )
    eval_dataset = TokenizedDataset(
        val_input, tokenizer, max_length=data_args.max_sequence_len, label_mapper=label_mapper
    )

    print('{:>5,} training samples'.format(len(train_dataset)))
    print('{:>5,} validation samples'.format(len(eval_dataset)))

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,  # for datacollator with padding
    )

    # Detecting last checkpoint.
    last_checkpoint = get_last_checkpoint_with_asserts(training_args)

    # Training
    if training_args.do_train:
        checkpoint = model_name_or_checkpoint(last_checkpoint, model_args)
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(
                train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        print("*** Evaluate ***")

        preds, labels, metrics = trainer.predict(eval_dataset)

        max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(
            eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

        trainer.log_metrics("post-training eval", metrics)
        trainer.save_metrics("post-training eval", metrics)

        prediction_output = []
        for preds_doc, labels_doc in zip(preds, labels):
            preds_doc = preds_doc[preds_doc != -100]
            labels_doc = labels_doc[labels_doc != -100]
            prediction_output.append([
                {'pred': float(p),
                 'label': float(l)}
                for p,l in zip(preds_doc, labels_doc)
            ])
        with open(os.path.join(training_args.output_dir, 'prediction_output.jsonl'), 'w') as f:
            jsonlines.Writer(f).write_all(prediction_output)



