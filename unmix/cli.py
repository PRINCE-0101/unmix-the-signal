"""Command-line interface."""
import argparse
import json
from .pipeline import benchmark, run_experiment


def main():
    parser = argparse.ArgumentParser(description="Separate mixed signals with from-scratch FastICA")
    parser.add_argument("--output", default="results", help="output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--benchmark", action="store_true", help="run 15 repeatable experiments")
    args = parser.parse_args()
    result = benchmark(args.output) if args.benchmark else run_experiment(args.output, args.seed)
    print(json.dumps(result, indent=2))

