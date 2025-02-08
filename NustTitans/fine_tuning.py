#!/usr/bin/env python
# coding=utf-8
import transformers,torch
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
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from datasets import Features, Value, ClassLabel, Dataset

os.environ['WANDB_DISABLED'] = 'true'

# Ensure required version of datasets library
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

logger = logging.getLogger(__name__)

import re
import string

def preprocess_text(text):
    """
    Preprocess the input text by removing emojis, punctuation, extra spaces, etc.
    """

    # Remove emojis using regex (Unicode characters for emojis)
    emoji_pattern = re.compile("[\U0001F600-\U0001F64F"
                               "\U0001F300-\U0001F5FF"
                               "\U0001F680-\U0001F6FF"
                               "\U0001F700-\U0001F77F"
                               "\U0001F780-\U0001F7FF"
                               "\U0001F800-\U0001F8FF"
                               "\U0001F900-\U0001F9FF"
                               "\U0001FA00-\U0001FA6F"
                               "\U0001FA70-\U0001FAFF"
                               "\U00002702-\U000027B0"
                               "\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    text = re.sub(emoji_pattern, '', text)  # Remove emojis

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)

    # Remove mentions (e.g., @username)
    text = re.sub(r"@\w+", '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove extra whitespace (e.g., multiple spaces, newlines)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


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
    max_train_samples: Optional[int] = field(
        default=None, metadata={"help": "For debugging purposes, truncate the number of training examples."}
    )
    max_eval_samples: Optional[int] = field(
        default=None, metadata={"help": "For debugging purposes, truncate the number of evaluation examples."}
    )
    max_predict_samples: Optional[int] = field(
        default=None, metadata={"help": "For debugging purposes, truncate the number of prediction examples."}
    )
@dataclass
class ModelArguments:

    model_name_or_path: str = field(default=None, metadata={"help": "Path to pretrained model or model identifier"})
    config_name: Optional[str] = field(default=None, metadata={"help": "Pretrained config name or path"})
    tokenizer_name: Optional[str] = field(default=None, metadata={"help": "Pretrained tokenizer name or path"})
    data_dir: Optional[str] = field(default=None, metadata={"help": "Path to dataset"})
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Directory to store the finetuned models"})
    do_lower_case: Optional[bool] = field(default=False, metadata={"help": "Lowercase tokenizer"})
    use_fast_tokenizer: bool = field(default=True, metadata={"help": "Use fast tokenizer"})
    model_revision: str = field(default="main", metadata={"help": "Model version"})
    use_auth_token: bool = field(default=False, metadata={"help": "Use auth token for private models"})
    ignore_mismatched_sizes: bool = field(default=False, metadata={"help": "Enable loading model with mismatched head sizes"})
    lr: Optional[float] = field(default=2e-5, metadata={"help": "Learning rate for training"})
    epochs: Optional[int] = field(default=3, metadata={"help": "Number of training epochs"})
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
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name or model_args.model_name_or_path, use_fast=model_args.use_fast_tokenizer)
    # Load the fine-tuning dataset (for example, a sentiment analysis dataset)
    df = pd.read_csv(model_args.data_dir)  # Replace with your fine-tuning data
    # Convert labels to float32 for multi-label classification
    df['labels'] = df[['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']].astype('float32').values.tolist()
 # Multi-label column
    df['text'] = df['text'].apply(preprocess_text)  # Apply text preprocessing

    # Convert dataset into Hugging Face Dataset format
    from datasets import Dataset
    fine_tuning_dataset = Dataset.from_pandas(df)
    # Ensure labels are of float32 type
    fine_tuning_dataset = fine_tuning_dataset.map(
        lambda x: {'labels': torch.tensor(x['labels'], dtype=torch.float32)},
        batched=True
    )

    # Preprocessing function for fine-tuning (same tokenizer)
    def fine_tune_preprocess(examples):
        return tokenizer(examples['text'], padding=True, truncation=True, max_length=128)

    # Tokenize the fine-tuning dataset
    fine_tuning_dataset = fine_tuning_dataset.map(fine_tune_preprocess, batched=True)

    # Convert to PyTorch format (input_ids, attention_mask, labels)
    #fine_tuning_dataset = fine_tuning_dataset.rename_column("label", "labels")
    fine_tuning_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    data_collator = DataCollatorWithPadding(tokenizer)
    # Load the model for fine-tuning
    model_for_fine_tuning = AutoModelForSequenceClassification.from_pretrained(
        num_labels=6,  # For multi-label classification (number of labels)
        problem_type="multi_label_classification",  # Multi-label classification setup
        pretrained_model_name_or_path = model_args.model_name_or_path,

    )

    # Fine-tuning training arguments
    fine_tuning_args = TrainingArguments(
        output_dir=training_args.output_dir,
        evaluation_strategy="epoch",
        learning_rate=model_args.lr,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=model_args.epochs,
        weight_decay=0.01,
        save_strategy="no",

        
    )

    # Set up Trainer for fine-tuning
    trainer_fine_tune = Trainer(
        model=model_for_fine_tuning,
        args=fine_tuning_args,
        train_dataset=fine_tuning_dataset,
        eval_dataset=fine_tuning_dataset,  # You can use separate validation set
        data_collator = data_collator,
    )

    # Start fine-tuning the model
    trainer_results = trainer_fine_tune.train()
    metrics = trainer_results.metrics
    max_train_samples = (data_args.max_train_samples if data_args.max_train_samples is not None else len(fine_tuning_dataset))
    metrics["train_samples"] = min(max_train_samples, len(fine_tuning_dataset))

    trainer_fine_tune.save_model(training_args.output_dir)  # Saves the tokenizer too for easy upload
    tokenizer.save_pretrained(training_args.output_dir)
    trainer_fine_tune.log_metrics("train", metrics)
    trainer_fine_tune.save_metrics("train", metrics)
    trainer_fine_tune.save_state()



if __name__ == "__main__":
    main()
