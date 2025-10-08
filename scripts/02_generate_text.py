# 02_generate_text.py
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

def generate_text(model_path, output_path="output/generated_text.txt"):
    """
    Generates text using the fine-tuned model.
    """
    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        torch_dtype=torch.float16
    ).to("cuda")

    # Load the fine-tuned adapter using PeftModel
    model = PeftModel.from_pretrained(
        base_model,
        model_path,
        torch_dtype=torch.float16
    ).to("cuda")

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")

    # Define a prompt
    messages = [
        {"role": "system", "content": "You are a reddit user making engaging and drama-filled content. Write in first person point of view as if you are writing on a social media platform."},
        {"role": "user", "content": "Write a short story asking reddit about whether you are an asshole for selling a TV that my brother uses."}
    ]

    # Format the prompt
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)

    # Encode the input
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to("cuda")

    # Set the `max_seq_length` attribute for the model
    model.base_model.config.max_seq_length = model.base_model.config.max_position_embeddings

    # Generate text
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=350,
        temperature=1.0,
        top_p=0.9,
        do_sample=True
    )

    # Decode and clean the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    start_index = generated_text.find("assistant") + len("assistant")
    cleaned_text = generated_text[start_index:].strip()

    # Save the generated text
    with open(output_path, "w") as f:
        f.write(cleaned_text)
    
    print(f"Generated text saved to {output_path}")
    return cleaned_text

if __name__ == "__main__":
    # Assuming the model is saved in the default location from the training script
    final_model_path = "/content/smollm2_finetuned/final_model"
    generate_text(final_model_path)
