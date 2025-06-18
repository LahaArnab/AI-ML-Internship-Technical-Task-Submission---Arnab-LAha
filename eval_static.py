import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from rouge_score import rouge_scorer
import pandas as pd

# --- Model Configuration ---
BASE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PEFT_MODEL_PATH = "tinyllama-colorist-v1/checkpoint-250"

# --- Evaluation Prompts ---
TEST_PROMPTS = [
    "Create a new Git branch and switch to it.",
    "Compress the folder reports into reports.tar.gz.",
    "List all Python files in the current directory recursively.",
    "Set up a virtual environment and install requests.",
    "Fetch only the first ten lines of a file named output.log.",
    "Clone a GitHub repo and list all Python files.",
    "Find all images in the current directory.",
    "Check GPU status on a Linux server.",
    "Create a virtual environment with Python 3.9.",
    "Delete all .log files older than 7 days.",
]

# Edge cases
EDGE_CASES = [
    "Delete the 'temp_files' folder and then create a new git branch called 'feature/new-ui'.",
    "Check the python version, then write 'Hello World' to a file named 'output.txt'.",
]

# Expected outputs for scoring
REFERENCE_PLANS = {
    TEST_PROMPTS[0]: "git checkout -b <branch_name>",
    TEST_PROMPTS[1]: "tar -czvf reports.tar.gz reports/",
    TEST_PROMPTS[2]: "find . -type f -name '*.py'",
    TEST_PROMPTS[3]: "python -m venv env\nsource env/bin/activate\npip install requests",
    TEST_PROMPTS[4]: "head -n 10 output.log",
    TEST_PROMPTS[5]: "git clone <repo_url>\nfind <repo_dir> -type f -name '*.py'",
    TEST_PROMPTS[6]: "find . -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \)",
    TEST_PROMPTS[7]: "nvidia-smi",
    TEST_PROMPTS[8]: "python3.9 -m venv env39\nsource env39/bin/activate",
    TEST_PROMPTS[9]: "find . -type f -name '*.log' -mtime +7 -delete",
    EDGE_CASES[0]: "rm -rf temp_files/\ngit checkout -b feature/new-ui",
    EDGE_CASES[1]: "python --version\necho 'Hello World' > output.txt",
}


def load_models_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    if not os.path.exists(PEFT_MODEL_PATH):
        raise FileNotFoundError(f"PEFT model path not found: {PEFT_MODEL_PATH}")
    peft_model = PeftModel.from_pretrained(base_model, PEFT_MODEL_PATH, from_transformers=True, device_map="auto")
    tuned_model = peft_model.merge_and_unload()
    return base_model, tuned_model, tokenizer


def formatted_prompt(question: str) -> str:
    return f"<|user|>\n{question}</s>\n<|assistant|>"


def generate(model, tokenizer, prompt_text: str) -> str:
    prompt = formatted_prompt(prompt_text)
    inputs = tokenizer(prompt, return_tensors="pt").to(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    gen_config = GenerationConfig(max_new_tokens=100, pad_token_id=tokenizer.eos_token_id,
                                  do_sample=False, repetition_penalty=1.2)
    outputs = model.generate(**inputs, generation_config=gen_config)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("<|assistant|>")[-1].strip()


def score_plan_quality(generated_plan: str, reference_plan: str) -> int:
    gen_steps = set(s.strip().lower() for s in generated_plan.split('\n') if s.strip())
    ref_steps = set(s.strip().lower() for s in reference_plan.split('\n') if s.strip())
    if gen_steps == ref_steps:
        return 2
    if gen_steps & ref_steps:
        return 1
    return 0


def main():
    try:
        base_model, tuned_model, tokenizer = load_models_and_tokenizer()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    results = []
    all_prompts = TEST_PROMPTS + EDGE_CASES

    for prompt in all_prompts:
        if prompt not in REFERENCE_PLANS:
            print(f"Skipping prompt (no reference): {prompt}")
            continue
        ref = REFERENCE_PLANS[prompt]
        base_out = generate(base_model, tokenizer, prompt)
        tuned_out = generate(tuned_model, tokenizer, prompt)
        base_rouge = scorer.score(ref, base_out)['rougeL'].fmeasure
        tuned_rouge = scorer.score(ref, tuned_out)['rougeL'].fmeasure
        base_quality = score_plan_quality(base_out, ref)
        tuned_quality = score_plan_quality(tuned_out, ref)
        results.append({
            "Prompt": prompt,
            "Reference": ref,
            "Base Output": base_out,
            "Base ROUGE-L": base_rouge,
            "Base Quality": base_quality,
            "Tuned Output": tuned_out,
            "Tuned ROUGE-L": tuned_rouge,
            "Tuned Quality": tuned_quality,
        })
        print(f"Prompt: {prompt}\nBase: {base_out}\nTuned: {tuned_out}\n")

    df = pd.DataFrame(results)
    print("\n--- Evaluation Summary ---")
    # Fallback to to_string to avoid tabulate dependency
    print(df.to_string(index=False))

    summary = pd.DataFrame({
        "Metric": ["Average ROUGE-L", "Average Quality"],
        "Base Model": [df['Base ROUGE-L'].mean(), df['Base Quality'].mean()],
        "Tuned Model": [df['Tuned ROUGE-L'].mean(), df['Tuned Quality'].mean()]
    })
    print("\n--- Average Scores ---")
    print(summary.to_string(index=False))

    df.to_csv("evaluation_results.csv", index=False)
    print("Results saved to evaluation_results.csv")

if __name__ == "__main__":
    main()