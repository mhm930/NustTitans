#!/usr/bin/env python
# coding=utf-8
import transformers, torch
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional
import datasets
import numpy as np
import pandas as pd
from datasets import load_dataset
import evaluate
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    AutoModelForMaskedLM,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from datasets import Features, Value, ClassLabel, Dataset
from transformers import DataCollatorForLanguageModeling

# Ensure required version of datasets library
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

logger = logging.getLogger(__name__)

@dataclass
class DataTrainingArguments:
    max_seq_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated."
        }
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={"help": "Whether to pad all samples to `max_seq_length`."}
    )

    domain_corpus: Optional[str] = field(
        default=None, metadata={"help": "Path to domain-specific corpus for adaptive pre-training"}
    )


@dataclass
class ModelArguments:

    model_name_or_path: str = field(default=None, metadata={"help": "Path to pretrained model or model identifier"})
    config_name: Optional[str] = field(default=None, metadata={"help": "Pretrained config name or path"})
    tokenizer_name: Optional[str] = field(default=None, metadata={"help": "Pretrained tokenizer name or path"})
    data_dir: Optional[str] = field(default=None, metadata={"help": "Path to dataset"})
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Directory to store the pretrained models"})
    do_lower_case: Optional[bool] = field(default=False, metadata={"help": "Lowercase tokenizer"})
    use_fast_tokenizer: bool = field(default=True, metadata={"help": "Use fast tokenizer"})
    model_revision: str = field(default="main", metadata={"help": "Model version"})
    use_auth_token: bool = field(default=False, metadata={"help": "Use auth token for private models"})
    ignore_mismatched_sizes: bool = field(default=False, metadata={"help": "Enable loading model with mismatched head sizes"})
    epochs: Optional[int] = field(default=3, metadata={"help": "Number of training epochs"})
    lr: Optional[float] = field(default=2e-5, metadata={"help": "Learning rate for training"})


def main():
    # Parse arguments from command line
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.setLevel(training_args.get_process_log_level())
    datasets.utils.logging.set_verbosity(training_args.get_process_log_level())
    transformers.utils.logging.set_verbosity(training_args.get_process_log_level())

    logger.info(f"Training/evaluation parameters {training_args}")

    # Handle last checkpoint detection
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint:
            logger.info(f"Checkpoint detected, resuming from {last_checkpoint}.")
        elif len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(f"Output directory ({training_args.output_dir}) is not empty.")

    # Set seed for reproducibility
    set_seed(training_args.seed)

    num_labels = 6

    # Model configuration
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name or model_args.model_name_or_path, use_fast=model_args.use_fast_tokenizer)
    model = AutoModelForMaskedLM.from_pretrained(model_args.model_name_or_path)

    def preprocess_function(examples):
        label_list = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
        texts = (examples['text'],)
        result = tokenizer(*texts, padding="max_length", max_length=data_args.max_seq_length, truncation=True)
        if "label" in examples:
            result["label"] = [label_list.index(l) for l in examples["label"]]
        return result

    if data_args.domain_corpus:
        # Metric setup
        metric = evaluate.load("accuracy")

        def compute_metrics(p: EvalPrediction):
            preds = np.argmax(p.predictions, axis=1)
            return metric.compute(predictions=preds, references=p.label_ids)

        # Load your domain-specific corpus
        domain_corpus = load_dataset(data_args.domain_corpus)  # Use the `domain_corpus` argument

        # Tokenization for MLM
        def preprocess_function(examples):
            result = tokenizer(examples['text'], padding=True, truncation=True, max_length=data_args.max_seq_length)
            return result

        # Split domain_corpus into 'train' and 'test'
        domain_corpus = domain_corpus.remove_columns("label")
        domain_corpus = domain_corpus["train"].train_test_split(train_size=0.9)
        domain_corpus = domain_corpus["test"].train_test_split(test_size=0.3)
        train_dataset = domain_corpus.map(preprocess_function, batched=True)["train"]
        eval_dataset =  domain_corpus.map(preprocess_function, batched=True)["test"]

        # Setup Data Collator for MLM
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=0.15,  # 15% of the tokens will be masked during training
        )

        # Fine-tuning the model on domain-specific corpus (Adaptive Pre-Training)
        domain_train_args = TrainingArguments(
            output_dir=training_args.output_dir,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            num_train_epochs=model_args.epochs,  # Using command-line argument for epochs
            save_steps=1,
            logging_steps=1,
            evaluation_strategy="steps",
            logging_dir="./logs",
            save_total_limit=1,
            disable_tqdm=True,
            report_to="none",
            save_strategy="no",
            learning_rate=model_args.lr,  # Using command-line argument for learning rate
        )

        trainer = Trainer(
            model=model,
            args=domain_train_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        # Start training
        train_result = trainer.train()
        print("Done adaptive pre-training!")
        metrics = train_result.metrics
        trainer.save_model(training_args.output_dir)  # Saves the tokenizer too for easy upload



if __name__ == "__main__":
    main()
