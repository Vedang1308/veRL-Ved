#!/usr/bin/env python3
"""
Test / Evaluate a model checkpoint on GSM8K math problems.
Displays step-by-step reasoning and checks if the final answer matches ground truth.
"""

import argparse
import os
import re
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SAMPLE_PROBLEMS = [
    {
        "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "ground_truth": "72"
    },
    {
        "question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 5 hours of babysitting. How much did she earn?",
        "ground_truth": "60"
    },
    {
        "question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
        "ground_truth": "5"
    },
    {
        "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
        "ground_truth": "3"
    },
    {
        "question": "Josh decides to try flipping a house. He buys a house for $80,000 and puts $50,000 into repairs. A year later, he sells it for $150,000. What was his profit?",
        "ground_truth": "20000"
    }
]


def extract_answer(text: str) -> str:
    """Extract number following #### or the last number in text."""
    match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")
    numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return ""


def main():
    parser = argparse.ArgumentParser(description="Test model on GSM8K problems")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model directory or Hugging Face model id (e.g., path to checkpoint's huggingface subfolder)")
    parser.add_argument("--data_file", type=str, default=None,
                        help="Optional path to test.parquet dataset")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of problems to test")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # If the user passed the main checkpoint folder, check if 'huggingface' subfolder exists
    model_dir = args.model_path
    if os.path.isdir(os.path.join(model_dir, "huggingface")):
        model_dir = os.path.join(model_dir, "huggingface")
        print(f"Using exported Hugging Face weights found at: {model_dir}")

    print(f"Loading tokenizer and model from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=args.device,
        trust_remote_code=True
    )
    model.eval()

    # Load problems
    problems = []
    if args.data_file and os.path.exists(args.data_file):
        try:
            import pandas as pd
            df = pd.read_parquet(args.data_file)
            for _, row in df.head(args.num_samples).iterrows():
                if "messages" in row:
                    msgs = row["messages"]
                    q = msgs[0]["content"]
                    ans = msgs[1]["content"]
                    gt = extract_answer(ans)
                    problems.append({"question": q, "ground_truth": gt})
        except Exception as e:
            print(f"Notice: Could not read {args.data_file} ({e}). Falling back to sample problems.")

    if not problems:
        problems = SAMPLE_PROBLEMS[:args.num_samples]

    instruction = "Let's think step by step and output the final answer after \"####\"."
    correct = 0

    print("\n" + "=" * 75)
    print(" EVALUATION ON GSM8K MATH PROBLEMS")
    print("=" * 75)

    for idx, item in enumerate(problems, 1):
        q = item["question"]
        if instruction not in q:
            prompt_text = f"{q} {instruction}"
        else:
            prompt_text = q

        messages = [
            {"role": "user", "content": prompt_text}
        ]

        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            input_text = f"User: {prompt_text}\nAssistant:"

        inputs = tokenizer(input_text, return_tensors="pt").to(args.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.0,  # Greedy for reproducible evaluation
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode generated text only
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        pred = extract_answer(response)
        gt = item["ground_truth"]
        is_match = (pred == gt)
        if is_match:
            correct += 1

        print(f"\n--- Problem {idx} ---")
        print(f"Question:      {q}")
        print(f"Ground Truth:  {gt}")
        print(f"Predicted:     {pred} [{'CORRECT' if is_match else 'INCORRECT'}]")
        print(f"Model Reasoning:\n{response}")
        print("-" * 75)

    accuracy = (correct / len(problems)) * 100
    print(f"\nSummary: {correct}/{len(problems)} correct ({accuracy:.1f}% accuracy)")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
