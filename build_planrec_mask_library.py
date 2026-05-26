#!/usr/bin/env python3
import argparse

from opal_llm.mask_library import (
    DEFAULT_MASK_STRATEGIES,
    generate_mask_library,
    save_mask_library,
)


def parse_csv_ints(value: str):
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def main():
    parser = argparse.ArgumentParser(description="Build OPAL-LLM decoder layer mask candidates.")
    parser.add_argument("--num_layers", type=int, required=True)
    parser.add_argument("--budgets", type=str, required=True, help="Comma-separated retained-layer budgets, e.g. 14,18,21,24,28")
    parser.add_argument(
        "--strategies",
        type=str,
        default=",".join(DEFAULT_MASK_STRATEGIES),
        help="Comma-separated strategies: uniform,first_k,last_k,ends_heavy,middle_heavy,shortgpt",
    )
    parser.add_argument("--random_masks_per_budget", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    masks = generate_mask_library(
        num_layers=args.num_layers,
        budgets=parse_csv_ints(args.budgets),
        strategies=[part.strip() for part in args.strategies.split(",") if part.strip()],
        random_masks_per_budget=args.random_masks_per_budget,
        seed=args.seed,
    )
    save_mask_library(args.output, masks)
    print(f"Saved {len(masks)} masks to {args.output}")


if __name__ == "__main__":
    main()
