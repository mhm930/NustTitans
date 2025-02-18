The main code is SentimentAnalysis.ipynb
As the training is done on kaggle, and the notebook is designed to have a command line capibility , where we can easily fine tune the parameters, without 
to look in the code everytime. Hence, in the notenotebook, it is first creating a directory "NustTitans" then it is writing two files:

**1. Adaptive_pretraining.py**
  **Arguments**
  
  **Model Arguments (ModelArguments)**
  
  model_name_or_path: Path to the pre-trained model (e.g., "bert-base-uncased").
  config_name (optional): Path to model configuration.
  tokenizer_name (optional): Path to tokenizer.
  epochs: Number of epochs to train the model (default: 3).
  lr: Learning rate for training (default: 2e-5).
  do_lower_case: Whether to lowercase the input text (default: False).
  
  **Data Arguments (DataTrainingArguments)**
  
  max_seq_length: Maximum input sequence length after tokenization (default: 128).
  overwrite_cache: Whether to overwrite cached datasets (default: False).
  pad_to_max_length: Whether to pad sequences to max_seq_length (default: True).
  domain_corpus: Path to the domain-specific dataset for adaptive pre-training.
  Training Arguments (TrainingArguments)
  output_dir: Directory where the trained model will be saved.
  per_device_train_batch_size: Batch size for training (default: 8).
  per_device_eval_batch_size: Batch size for evaluation (default: 16).
  num_train_epochs: Number of epochs for training.
  learning_rate: Learning rate for training.

  python adaptive_pretraining.py \
    --model_name_or_path <path_to_the_model>\
    --domain_corpus <path_to_domain_corpus> \
    --output_dir <output_directory> \
    --epochs 3 \
    --lr 2e-5

**2. FineTuning.py**

**ModelArguments:**

model_name_or_path: The name of the pre-trained model (e.g., bert-base-uncased).
lr: The learning rate for training.
epochs: The number of epochs for fine-tuning.
DataTrainingArguments:
max_seq_length: Maximum length of input sequences (after tokenization).
max_train_samples, max_eval_samples: Limit the number of samples used for training and evaluation (for debugging).

**Model Training and Evaluation:**

The Trainer is set up with the fine_tuning_args, which specify how to train the model, including batch sizes, evaluation strategy, and saving configuration.
The model is fine-tuned with the provided dataset for multi-label classification and evaluated on the same dataset (a separate validation set can be used if available).

**Output:**

Model: The trained model is saved to the output directory (output_dir).
Tokenizer: The tokenizer is saved along with the model for later use.
Metrics: Training metrics are logged and saved for later analysis.

**Fine_tuning**

!python /kaggle/working/NustTitans/fine_tuning.py \
  --model_name_or_path '<pretrained_model_path>'  \
  --output_dir '<model_save_path>' \ 
  --data_dir '<path_to_csv_file>' \
  --do_train \
  --per_device_train_batch_size 8 \
  --lr 5e-6 \
  --epochs 20 \
  --max_seq_length 128 \
  --save_steps -10               



