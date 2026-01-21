#!/usr/bin/env python3
"""
Simple HumanEval runner for CAD coder models
"""

import json
import subprocess
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Install human-eval if not available
try:
    from human_eval.data import read_problems, write_jsonl
    from human_eval.evaluation import evaluate_functional_correctness
except ImportError:
    print("Installing human-eval...")
    subprocess.run(["pip", "install", "human-eval"], check=True)
    from human_eval.data import read_problems, write_jsonl
    from human_eval.evaluation import evaluate_functional_correctness


class HumanEvalRunner:
    def __init__(self, model_name_or_path: str, tokenizer_name: str = None):
        """
        Initialize HumanEval runner

        Args:
            model_name_or_path: Path to model or HuggingFace model name
            tokenizer_name: Tokenizer name (defaults to model_name_or_path)
        """
        self.model_name = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name or model_name_or_path
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=torch.float16, device_map="auto"
        )

        # Set pad token if not available
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_completion(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.95,
    ) -> str:
        """Generate code completion for a given prompt"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the generated part
        generated = outputs[0][inputs["input_ids"].shape[1] :]
        completion = self.tokenizer.decode(generated, skip_special_tokens=True)

        # Clean up completion (stop at function end or next def)
        completion = self._clean_completion(completion)
        return completion

    def _clean_completion(self, completion: str) -> str:
        """Clean up generated completion"""
        lines = completion.split("\n")
        cleaned_lines = []

        for line in lines:
            # Stop at next function definition
            if line.strip().startswith("def ") and cleaned_lines:
                break
            # Stop at class definition
            if line.strip().startswith("class "):
                break
            cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    def run_humaneval(
        self,
        num_samples: int = 1,
        output_file: str = None,
        max_problems: Optional[int] = None,
        start_problem: int = 0,
        specific_problems: Optional[List[str]] = None,
    ) -> Dict:
        """
        Run HumanEval benchmark

        Args:
            num_samples: Number of samples per problem
            output_file: File to save results (optional)
            max_problems: Maximum number of problems to run (optional)
            start_problem: Starting problem index (0-based)
            specific_problems: List of specific problem IDs to run (optional)

        Returns:
            Dictionary with evaluation results
        """
        all_problems = read_problems()

        # Filter problems based on parameters
        if specific_problems:
            problems = {k: v for k, v in all_problems.items() if k in specific_problems}
            print(f"Running on {len(problems)} specific problems: {specific_problems}")
        else:
            problems_list = list(all_problems.items())

            # Apply start_problem offset
            if start_problem > 0:
                problems_list = problems_list[start_problem:]
                print(f"Starting from problem {start_problem}")

            # Apply max_problems limit
            if max_problems:
                problems_list = problems_list[:max_problems]
                print(f"Limited to {max_problems} problems")

            problems = dict(problems_list)

        samples = []

        print(
            f"Running HumanEval on {len(problems)} problems (out of {len(all_problems)} total)..."
        )
        print(f"Model: {self.model_name}")
        print(f"Samples per problem: {num_samples}")

        for i, (task_id, problem) in enumerate(problems.items()):
            print(f"Problem {i+1}/{len(problems)}: {task_id}")

            prompt = problem["prompt"]

            for sample_idx in range(num_samples):
                completion = self.generate_completion(prompt)
                samples.append({"task_id": task_id, "completion": completion})

        # Save samples
        if output_file is None:
            output_file = f"humaneval_samples_{self.model_name.replace('/', '_')}.jsonl"

        write_jsonl(output_file, samples)
        print(f"Saved {len(samples)} samples to {output_file}")

        # Evaluate
        print("Evaluating functional correctness...")
        results = evaluate_functional_correctness(output_file)

        print("=" * 50)
        print("HumanEval Results:")
        print("=" * 50)
        for k, v in results.items():
            print(f"{k}: {v:.3f}")

        return results


def main():
    """Example usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Run HumanEval on a model")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument(
        "--tokenizer", type=str, default=None, help="Tokenizer name (defaults to model)"
    )
    parser.add_argument(
        "--samples", type=int, default=1, help="Number of samples per problem"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output file for samples"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.2, help="Generation temperature"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=512, help="Max new tokens to generate"
    )
    parser.add_argument(
        "--max_problems",
        type=int,
        default=None,
        help="Maximum number of problems to run",
    )
    parser.add_argument(
        "--start_problem", type=int, default=0, help="Starting problem index (0-based)"
    )
    parser.add_argument(
        "--problems",
        type=str,
        nargs="+",
        default=None,
        help="Specific problem IDs to run (e.g., HumanEval/0 HumanEval/1)",
    )

    args = parser.parse_args()

    # Initialize runner
    runner = HumanEvalRunner(args.model, args.tokenizer)

    # Run evaluation
    results = runner.run_humaneval(
        num_samples=args.samples,
        output_file=args.output,
        max_problems=args.max_problems,
        start_problem=args.start_problem,
        specific_problems=args.problems,
    )

    # Save results
    results_file = f"humaneval_results_{args.model.replace('/', '_')}.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "model": args.model,
                "samples_per_problem": args.samples,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
