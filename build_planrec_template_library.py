#!/usr/bin/env python3
import argparse

from opal_llm.template_library import (
    DEFAULT_TEMPLATE_STRATEGIES,
    generate_template_library,
    save_template_library,
)


def parse_csv_ints(value: str):
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def main():
    parser = argparse.ArgumentParser(description="Build a finite OPAL-LLM decoder layer template library.")
    parser.add_argument("--num_layers", type=int, required=True)
    parser.add_argument("--budgets", type=str, required=True, help="Comma-separated retained-layer budgets, e.g. 14,18,21,24,28")
    parser.add_argument(
        "--strategies",
        type=str,
        default=",".join(DEFAULT_TEMPLATE_STRATEGIES),
        help="Comma-separated strategies: uniform,first_k,last_k,ends_heavy,middle_heavy",
    )
    parser.add_argument("--random_templates_per_budget", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    templates = generate_template_library(
        num_layers=args.num_layers,
        budgets=parse_csv_ints(args.budgets),
        strategies=[part.strip() for part in args.strategies.split(",") if part.strip()],
        random_templates_per_budget=args.random_templates_per_budget,
        seed=args.seed,
    )
    save_template_library(args.output, templates)
    print(f"Saved {len(templates)} templates to {args.output}")


if __name__ == "__main__":
    main()
