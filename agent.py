import argparse
import json
import os
import subprocess
from time import perf_counter

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# --- Model Configuration ---
# Base model identifier
BASE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# Path to your fine-tuned PEFT model checkpoint
PEFT_MODEL_PATH = "tinyllama-colorist-v1/checkpoint-250"
# Path for logging
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "trace.jsonl")

def setup_logging():
    """Create the logging directory if it doesn't exist."""
    os.makedirs(LOG_DIR, exist_ok=True)

def load_model():
    """
    Loads the base model and applies the PEFT adapter.
    Returns the merged model and the tokenizer.
    """
    # print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # print(f"Loading PEFT adapter from {PEFT_MODEL_PATH}...")
    # Ensure the adapter path exists
    if not os.path.exists(PEFT_MODEL_PATH):
        raise FileNotFoundError(
            f"PEFT model path not found: {PEFT_MODEL_PATH}. "
            "Please ensure the path is correct and the checkpoint files are present."
        )

    peft_model = PeftModel.from_pretrained(model, PEFT_MODEL_PATH, from_transformers=True, device_map="auto")
    
    # print("Merging model and adapter...")
    model = peft_model.merge_and_unload()
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token # Set pad token

    # print("Model loaded successfully.")
    return model, tokenizer

def formatted_prompt(question: str) -> str:
    """Formats the user's question into the expected prompt structure."""
    return f"<|user|>\n{question}</s>\n<|assistant|>"

def generate_plan(model, tokenizer, user_input):
    """
    Generates a step-by-step plan using the fine-tuned model.
    
    Args:
        model: The loaded Hugging Face model.
        tokenizer: The loaded Hugging Face tokenizer.
        user_input (str): The natural language instruction from the user.

    Returns:
        str: The generated plan as a string.
    """
    prompt = formatted_prompt(user_input)
    
    # Use the same generation config as in your notebook for consistency
    generation_config = GenerationConfig(
        penalty_alpha=0.6,
        do_sample=True,
        top_k=5,
        temperature=0.5,
        repetition_penalty=1.2,
        max_new_tokens=100,  # Increased tokens for more detailed plans
        pad_token_id=tokenizer.eos_token_id
    )
    
    print("\nGenerating plan...")
    start_time = perf_counter()
    
    inputs = tokenizer(prompt, return_tensors="pt").to('cuda' if torch.cuda.is_available() else 'cpu')
    
    outputs = model.generate(**inputs, generation_config=generation_config)
    plan_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Clean up the output to only show the assistant's response
    plan_text = plan_text.split("<|assistant|>")[-1].strip()

    output_time = perf_counter() - start_time
    print(f"Time taken for inference: {round(output_time, 2)} seconds")
    
    return plan_text

def process_and_log_plan(instruction, plan):
    """
    Processes the generated plan: logs it and performs a dry-run of the first command.
    """
    steps = [step.strip() for step in plan.split('\n') if step.strip()]
    
    if not steps:
        print("Generated plan was empty.")
        return

    print("\n--- Generated Plan ---")
    print(plan)
    print("----------------------\n")

    # Log the entire interaction
    log_entry = {
        "instruction": instruction,
        "generated_plan": steps
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    print(f"Plan logged to {LOG_FILE}")

    # --- Dry-Run Execution ---
    first_step = steps[0]
    # Simple heuristic to check if the first step is a shell command
    # This can be improved with more robust parsing.
    shell_keywords = ['git', 'mkdir', 'ls', 'cd', 'python', 'pip', 'touch', 'rm', 'mv', 'cp', 'echo']
    if any(first_step.startswith(keyword) for keyword in shell_keywords):
        print("--- Dry Run ---")
        dry_run_command = f"echo \"Executing: {first_step}\""
        print(f"$ {dry_run_command}")
        subprocess.run(dry_run_command, shell=True)
        print("---------------\n")
    else:
        print("First step does not appear to be a direct shell command. Skipping dry-run.\n")


def main():
    """Main function to run the CLI agent."""
    parser = argparse.ArgumentParser(description="CLI Agent that generates and logs plans from natural language.")
    parser.add_argument("instruction", type=str, help="The natural language instruction for the agent.")
    
    args = parser.parse_args()
    
    try:
        setup_logging()
        model, tokenizer = load_model()
        plan = generate_plan(model, tokenizer, args.instruction)
        process_and_log_plan(args.instruction, plan)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure your fine-tuned model checkpoint is available.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()