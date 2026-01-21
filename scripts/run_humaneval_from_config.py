#!/usr/bin/env python3
"""
Run HumanEval on the model specified in config.yaml
This script reads the cadcoder.model_name from config and evaluates it on HumanEval benchmark
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add the parent directory to the path to import modules
sys.path.append(str(Path(__file__).parent.parent))

# Install human-eval if not available
try:
    from human_eval.data import read_problems, write_jsonl
    from human_eval.evaluation import evaluate_functional_correctness

    HUMANEVAL_AVAILABLE = True
except ImportError:
    print("❌ human-eval not installed. Installing...")
    import subprocess

    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "human-eval"], check=True
        )
        from human_eval.data import read_problems, write_jsonl
        from human_eval.evaluation import evaluate_functional_correctness

        HUMANEVAL_AVAILABLE = True
        print("✅ human-eval installed successfully")
    except Exception as e:
        print(f"❌ Failed to install human-eval: {e}")
        HUMANEVAL_AVAILABLE = False


class ConfigBasedHumanEvalRunner:
    def __init__(self, config_path: str = None, override_model: str = None):
        """
        Initialize HumanEval runner from config file

        Args:
            config_path: Path to config.yaml file
            override_model: Model name to override config (optional)
        """
        if not HUMANEVAL_AVAILABLE:
            raise ImportError("human-eval package is required but not available")

        # Load config
        self.config = self._load_config(config_path)

        # Get model name from config or override
        self.model_name = override_model or self.config.get("cadcoder", {}).get(
            "model_name"
        )
        if not self.model_name:
            raise ValueError(
                "No model_name found in config.cadcoder.model_name or provided as override"
            )

        print(f"🚀 Initializing HumanEval runner for model: {self.model_name}")

        # Initialize model and tokenizer
        self._load_model()

    def _load_config(self, config_path: str = None) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if config_path is None:
            # Look for config.yaml in parent directory
            config_path = Path(__file__).parent.parent / "config.yaml"

        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        print(f"📄 Loading config from: {config_path}")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        return config

    def _load_model(self):
        """Load model and tokenizer from HuggingFace"""
        print(f"📥 Loading tokenizer: {self.model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        except Exception as e:
            print(f"❌ Failed to load tokenizer: {e}")
            raise

        # Set pad token if not available
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print("🔧 Set pad_token to eos_token")

        print(f"📥 Loading model: {self.model_name}")
        try:
            # Try to load with appropriate dtype and device mapping
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16
                if torch.cuda.is_available()
                else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,  # Some models might need this
            )
            print("✅ Model loaded successfully")
            print(f"🔧 Model device: {next(self.model.parameters()).device}")
            print(f"🔧 Model dtype: {next(self.model.parameters()).dtype}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise

    def generate_completion(
        self,
        prompt: str,
        max_new_tokens: int = None,
        temperature: float = None,
        top_p: float = 0.95,
        do_sample: bool = None,
    ) -> str:
        """Generate code completion for a given prompt"""
        # Use config values as defaults
        max_new_tokens = max_new_tokens or self.config.get("text2cad", {}).get(
            "max_tokens", 512
        )
        temperature = temperature or 0.2
        do_sample = do_sample if do_sample is not None else (temperature > 0)

        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.get("data", {}).get("max_total_len", 1024),
            )

            # Move inputs to model device
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                )

            # Decode only the generated part
            generated = outputs[0][inputs["input_ids"].shape[1] :]
            completion = self.tokenizer.decode(generated, skip_special_tokens=True)

            # Clean up completion
            completion = self._clean_completion(completion)
            return completion

        except Exception as e:
            print(f"❌ Error during generation: {e}")
            return ""

    def _clean_completion(self, completion: str) -> str:
        """Clean up generated completion to make it valid"""
        lines = completion.split("\n")
        cleaned_lines = []

        for line in lines:
            # Stop at next function definition
            if line.strip().startswith("def ") and cleaned_lines:
                break
            # Stop at class definition
            if line.strip().startswith("class "):
                break
            # Stop at import statements (usually means we've gone too far)
            if line.strip().startswith("import ") and cleaned_lines:
                break
            # Stop at certain markers that indicate end of function
            if line.strip() in ['"""', "'''"] and cleaned_lines:
                cleaned_lines.append(line)
                break

            cleaned_lines.append(line)

        result = "\n".join(cleaned_lines)

        # Remove any trailing incomplete lines
        result = result.rstrip()

        return result

    def run_humaneval(
        self,
        num_samples: int = 1,
        output_dir: Optional[str] = None,
        temperature: float = 0.2,
        save_detailed: bool = True,
        max_problems: Optional[int] = None,
        start_problem: int = 0,
        specific_problems: Optional[List[str]] = None,
    ) -> Dict:
        """
        Run HumanEval benchmark

        Args:
            num_samples: Number of samples per problem
            output_dir: Directory to save results
            temperature: Generation temperature
            save_detailed: Whether to save detailed results
            max_problems: Maximum number of problems to run (optional)
            start_problem: Starting problem index (0-based)
            specific_problems: List of specific problem IDs to run (optional)

        Returns:
            Dictionary with evaluation results
        """
        if output_dir is None:
            output_dir_path = Path(__file__).parent.parent / "results" / "humaneval"
        else:
            output_dir_path = Path(output_dir)

        output_dir_path.mkdir(parents=True, exist_ok=True)

        # Load problems
        all_problems = read_problems()

        # Filter problems based on parameters
        if specific_problems:
            problems = {k: v for k, v in all_problems.items() if k in specific_problems}
            print(
                f"🎯 Running on {len(problems)} specific problems: {specific_problems}"
            )
        else:
            problems_list = list(all_problems.items())

            # Apply start_problem offset
            if start_problem > 0:
                problems_list = problems_list[start_problem:]
                print(f"⏭️  Starting from problem {start_problem}")

            # Apply max_problems limit
            if max_problems:
                problems_list = problems_list[:max_problems]
                print(f"🔢 Limited to {max_problems} problems")

            problems = dict(problems_list)

        samples = []

        # Create timestamp for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_safe_name = self.model_name.replace("/", "_").replace("\\", "_")
        run_name = f"{model_safe_name}_{timestamp}"

        print("=" * 80)
        print("🧪 Starting HumanEval Evaluation")
        print("=" * 80)
        print(f"📋 Model: {self.model_name}")
        print(f"📊 Problems: {len(problems)}")
        print(f"🎯 Samples per problem: {num_samples}")
        print(f"🌡️  Temperature: {temperature}")
        print(f"📁 Output directory: {output_dir_path}")
        print("=" * 80)

        # Process each problem
        for i, (task_id, problem) in enumerate(problems.items()):
            print(f"🔄 Problem {i+1}/{len(problems)}: {task_id}")

            prompt = problem["prompt"]

            # Generate multiple samples for this problem
            for sample_idx in range(num_samples):
                if num_samples > 1:
                    print(f"   Sample {sample_idx+1}/{num_samples}")

                completion = self.generate_completion(prompt, temperature=temperature)

                samples.append({"task_id": task_id, "completion": completion})

                # Save detailed sample info if requested
                if save_detailed:
                    sample_info = {
                        "task_id": task_id,
                        "prompt": prompt,
                        "completion": completion,
                        "canonical_solution": problem.get("canonical_solution", ""),
                        "test": problem.get("test", ""),
                        "sample_index": sample_idx,
                        "timestamp": datetime.now().isoformat(),
                    }

                    # Save individual sample
                    sample_file = output_dir_path / f"{run_name}_samples_detailed.jsonl"
                    with open(sample_file, "a") as f:
                        f.write(json.dumps(sample_info) + "\n")

        # Save samples for evaluation
        samples_file = output_dir_path / f"{run_name}_samples.jsonl"
        write_jsonl(str(samples_file), samples)
        print(f"✅ Saved {len(samples)} samples to {samples_file}")

        # Evaluate functional correctness
        print("🔍 Evaluating functional correctness...")
        try:
            results = evaluate_functional_correctness(str(samples_file))
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            return {"error": str(e)}

        # Print results
        print("=" * 80)
        print("📊 HumanEval Results:")
        print("=" * 80)
        for k, v in results.items():
            print(f"{k}: {v:.4f}")
        print("=" * 80)

        # Save comprehensive results
        final_results = {
            "model_name": self.model_name,
            "config_used": self.config,
            "evaluation_params": {
                "num_samples": num_samples,
                "temperature": temperature,
                "timestamp": timestamp,
            },
            "results": results,
            "num_problems": len(problems),
            "total_samples": len(samples),
        }

        results_file = output_dir_path / f"{run_name}_results.json"
        with open(results_file, "w") as f:
            json.dump(final_results, f, indent=2)

        print(f"💾 Detailed results saved to {results_file}")

        return final_results


def main():
    """Main function with CLI interface"""
    parser = argparse.ArgumentParser(
        description="Run HumanEval on model from config.yaml"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml file (default: ../config.yaml)",
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Override model name from config"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help="Number of samples per problem (default: 1)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Generation temperature (default: 0.2)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Output directory for results"
    )
    parser.add_argument(
        "--no-detailed",
        action="store_true",
        help="Don't save detailed sample information",
    )
    parser.add_argument(
        "--max-problems",
        type=int,
        default=None,
        help="Maximum number of problems to run",
    )
    parser.add_argument(
        "--start-problem", type=int, default=0, help="Starting problem index (0-based)"
    )
    parser.add_argument(
        "--problems",
        type=str,
        nargs="+",
        default=None,
        help="Specific problem IDs to run (e.g., HumanEval/0 HumanEval/1)",
    )

    args = parser.parse_args()

    try:
        # Initialize runner
        runner = ConfigBasedHumanEvalRunner(
            config_path=args.config, override_model=args.model
        )

        # Run evaluation
        results = runner.run_humaneval(
            num_samples=args.samples,
            output_dir=args.output_dir,
            temperature=args.temperature,
            save_detailed=not args.no_detailed,
            max_problems=args.max_problems,
            start_problem=args.start_problem,
            specific_problems=args.problems,
        )

        if "error" not in results:
            print("🎉 Evaluation completed successfully!")
            print(f"🏆 Pass@1: {results['results'].get('pass@1', 'N/A'):.4f}")
        else:
            print(f"❌ Evaluation failed: {results['error']}")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
