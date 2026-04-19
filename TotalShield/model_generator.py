from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import os

# Step 1: Load base model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Step 2: Add padding token (GPT-2 doesn't have one by default)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Step 3: Save directory for final model
output_dir = "./TotalShieldModel"

# Step 4: Prepare a dummy safe dataset
with open("train.txt", "w") as f:
    f.write("This is a clean and secure sample.\n" * 100)

# Step 5: Load dataset using 🤗 Datasets
dataset = load_dataset("text", data_files={"train": "train.txt"})

def tokenize_function(examples):
    return tokenizer(examples["text"], return_special_tokens_mask=True, truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Step 6: Define TrainingArguments
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    save_strategy="epoch",
    save_total_limit=1,
    save_safetensors=True,  # enables .safetensors shards
    logging_steps=10,
    evaluation_strategy="no",
    report_to="none",
    fp16=False
)

# Step 7: Train with HuggingFace Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

# Step 8: Save everything
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"[✓] Model saved in {output_dir}")
