import os
from dataclasses import dataclass, field
from typing import Optional
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Enable CuDNN benchmarking for optimized CUDA operations
torch.backends.cudnn.benchmark = True

# Set a manual seed for reproducibility
torch.manual_seed(42)

@dataclass
class ScriptArguments:
    """
    These arguments can be adjusted based on your hardware and requirements.
    """
    # Training parameters
    per_device_train_batch_size: Optional[int] = field(
        default=2, metadata={"help": "Batch size per device during training."}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "Number of gradient accumulation steps."}
    )
    learning_rate: Optional[float] = field(
        default=3e-4, metadata={"help": "Initial learning rate for AdamW."}
    )
    max_seq_length: Optional[int] = field(
        default=256, metadata={"help": "Maximum sequence length for inputs."}
    )
    lora_alpha: Optional[int] = field(
        default=16, metadata={"help": "LoRA alpha parameter."}
    )
    lora_dropout: Optional[float] = field(
        default=0.05, metadata={"help": "LoRA dropout rate."}
    )
    lora_r: Optional[int] = field(
        default=8, metadata={"help": "LoRA rank parameter."}
    )
    max_steps: Optional[int] = field(
        default=1000, metadata={"help": "Total number of training steps to perform."}
    )

    # Model and dataset paths
    model_name: Optional[str] = field(
        default="gpt2",
        metadata={"help": "The model to train from the Hugging Face hub or local path."}
    )
    dataset_path: Optional[str] = field(
        default="df_latest.csv",
        metadata={"help": "Path to the preprocessed dataset CSV file."}
    )

    # Precision and optimization settings
    use_8bit: Optional[bool] = field(
        default=True,
        metadata={"help": "Use 8-bit quantization for the base model."},
    )
    fp16: Optional[bool] = field(
        default=True,
        metadata={"help": "Enable fp16 training."},
    )
    bf16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable bf16 training."},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False,  # Disabled for faster training
        metadata={"help": "Enable gradient checkpointing."},
    )
    optim: Optional[str] = field(
        default="adamw_torch",
        metadata={"help": "Optimizer to use."},
    )
    lr_scheduler_type: str = field(
        default="cosine",
        metadata={"help": "Learning rate scheduler type."},
    )
    warmup_steps: int = field(
        default=250, metadata={"help": "Warmup steps for LR scheduler."}
    )

    # Logging and saving
    output_dir: str = field(
        default="./gpt2-finetuned",
        metadata={"help": "Directory to save the model and checkpoints."},
    )
    save_steps: int = field(
        default=500, metadata={"help": "Save checkpoint every X steps."}
    )
    logging_steps: int = field(
        default=50, metadata={"help": "Log every X steps."}
    )
    report_to: Optional[str] = field(
        default="none",
        metadata={"help": "Reporting platform (e.g., 'wandb')."},
    )

# Initialize the argument parser and parse the arguments
parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

print("Starting dataset preparation...")

def create_prompt(example):
    """
    Creates a standardized prompt for each example in the dataset.
    """
    song_tags = example['tag']
    song_lyrics = example['lyrics']
    prompt = f"Song with the following genres: {song_tags}. Here are the song lyrics:\n{song_lyrics}"
    example['text'] = prompt
    return example

# Load the dataset using streaming to handle large datasets efficiently
print(f"Loading dataset from {script_args.dataset_path}...")
dataset = load_dataset('csv', data_files=script_args.dataset_path, streaming=True)
dataset = dataset['train']
print("Dataset loaded.")

# Apply the prompt creation function to each example
print("Applying prompt creation...")
dataset = dataset.map(create_prompt)
print("Prompt creation applied.")

# Remove unnecessary columns to save memory
print("Removing unnecessary columns...")
# Retrieve column names from the first example
try:
    first_example = next(iter(dataset))
    columns = list(first_example.keys())
    print(f"Columns in the dataset: {columns}")
except StopIteration:
    raise ValueError("Dataset is empty.")

columns_to_remove = [col for col in columns if col not in ['text']]
print(f"Columns to remove: {columns_to_remove}")
dataset = dataset.remove_columns(columns_to_remove)
print("Unnecessary columns removed.")

# Initialize the tokenizer
print("Initializing tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
print("Tokenizer initialized.")

def tokenize_function(examples):
    """
    Tokenizes the 'text' field in each example.
    Adds 'labels' field for causal language modeling.
    """
    tokens = tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=script_args.max_seq_length,
    )
    tokens['labels'] = tokens['input_ids'].copy()
    return tokens

# Tokenize the dataset
print("Tokenizing dataset...")
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=['text'],  # Remove the original text column after tokenization
)
print("Dataset tokenization complete.")

# Load the GPT-2 Small model with 8-bit quantization for memory efficiency
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    load_in_8bit=script_args.use_8bit,
    device_map="auto",
)
print("Model loaded.")

# Prepare the model for LoRA fine-tuning
print("Preparing model for LoRA...")
model = prepare_model_for_kbit_training(model)
print("Model prepared for LoRA.")

# Configure LoRA settings
print("Configuring LoRA...")
peft_config = LoraConfig(
    r=script_args.lora_r,
    lora_alpha=script_args.lora_alpha,
    target_modules=['c_attn', 'c_proj'],  # LoRA targets for GPT-2 Small
    lora_dropout=script_args.lora_dropout,
    bias='none',
    task_type="CAUSAL_LM",
)
print(f"LoRA configuration: {peft_config}")

# Apply LoRA configuration to the model
print("Applying LoRA configuration to model...")
model = get_peft_model(model, peft_config)
print("LoRA configuration applied.")

# Define training arguments
print("Defining training arguments...")
training_arguments = TrainingArguments(
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    max_steps=script_args.max_steps,  # Total training steps
    learning_rate=script_args.learning_rate,
    bf16=script_args.bf16,
    fp16=script_args.fp16,
    optim=script_args.optim,
    lr_scheduler_type=script_args.lr_scheduler_type,
    warmup_steps=script_args.warmup_steps,
    logging_steps=script_args.logging_steps,
    save_steps=script_args.save_steps,
    report_to=script_args.report_to,
    logging_dir=f"{script_args.output_dir}/logs",
    save_total_limit=2,
    gradient_checkpointing=script_args.gradient_checkpointing,  # Now False
    load_best_model_at_end=False,
    evaluation_strategy="no",
)
print(f"Training arguments defined.")

# Initialize the Trainer
print("Initializing trainer...")

trainer = Trainer(
    model=model,
    args=training_arguments,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=default_data_collator,
)
print("Trainer initialized.")

# Start the training process
print("Starting model training...")
trainer.train()
print("Model training complete.")

# Save the fine-tuned model and tokenizer
print(f"Saving model to {script_args.output_dir}...")
trainer.save_model(script_args.output_dir)
tokenizer.save_pretrained(script_args.output_dir)
print("Model saved successfully.")
