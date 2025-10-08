# 01_train_model.py
import torch
from transformers import TrainingArguments, Trainer, AutoTokenizer
from datasets import load_dataset
import wandb
from unsloth import FastLanguageModel
from peft import PeftModel
import os
from google.colab import drive

def train_model():
    """
    Fine-tunes a language model on the AITA stories dataset.
    """
    # Mount Google Drive
    drive.mount('/content/drive/')

    stories = '/content/drive/MyDrive/Personal_stuff/brainrot/aitah_stories.json'

    # Initialize Wandb
    wandb.login()
    wandb.init(
        project="demo-yt-video",
        config={
            "learning_rate": 5e-5,
            "architecture": "SmolLM2-1.7B-Instruct",
            "dataset": stories,
            "epochs": 3,
        }
    )

    # Load the correct tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")

    # Load the model with Unsloth
    model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    model, _ = FastLanguageModel.from_pretrained(
        model_name=model_name,
        dtype=torch.float16,
        load_in_4bit=True,
    )

    # Load the dataset
    dataset = load_dataset("json", data_files={"train": stories}, split="train")
    print("Dataset size:", len(dataset))

    # Split the dataset
    train_test_split = dataset.train_test_split(test_size=0.2)
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]

    # Tokenization function
    def tokenize_function(examples):
        combined_texts = [f"{prompt} {completion}" for prompt, completion in zip(examples["instruction"], examples["output"])]
        tokenized = tokenizer(combined_texts, truncation=True, max_length=512, padding="max_length", return_tensors="pt")
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized

    # Tokenize the datasets
    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)

    # Print a sample tokenized output for debugging
    sample_output = tokenized_train_dataset[0]
    print("Sample tokenized output:", sample_output)
    print("Decoded sample output:", tokenizer.decode(sample_output["input_ids"]))

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )

    # Clear GPU Cache
    print("Clearing GPU cache...")
    torch.cuda.empty_cache()

    # Define Training Arguments
    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir="/content/smollm2_finetuned",
        num_train_epochs=10,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        fp16=True,
        logging_steps=10,
        save_steps=30,
        eval_steps=30,
        save_total_limit=3,
        learning_rate=3e-5,
        logging_dir="./logs",
        report_to="wandb",
        run_name="SmolLM2_FineTuning_Experiment"
    )

    # Initialize the Trainer
    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
    )

    # Train the model
    trainer.train()
    
    # Save the final model
    final_model_path = "/content/smollm2_finetuned/final_model"
    trainer.save_model(final_model_path)
    print(f"Model saved to {final_model_path}")

if __name__ == "__main__":
    train_model()
