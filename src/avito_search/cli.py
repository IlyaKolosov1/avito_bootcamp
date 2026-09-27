"""Command-line entry point for reproducible experiments."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from avito_search.experiment import (
    compare_evaluations,
    evaluate_experiment,
    run_experiment,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="avito-experiment")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run one experiment")
    run_parser.add_argument("--config", required=True, help="path to TOML config")
    run_parser.add_argument(
        "--output-dir",
        default="runs",
        help="directory for predictions and metadata",
    )
    run_parser.add_argument(
        "--query-limit",
        type=int,
        default=None,
        help="optional query limit for a smoke test",
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="evaluate one experiment on proxy validation",
    )
    evaluate_parser.add_argument("--config", required=True, help="path to TOML config")
    evaluate_parser.add_argument(
        "--output-dir",
        default="runs",
        help="directory for predictions and metrics",
    )
    evaluate_parser.add_argument(
        "--query-limit",
        type=int,
        default=None,
        help="override validation query count for a smoke test",
    )

    compare_parser = subparsers.add_parser(
        "compare",
        help="compare all completed validation runs",
    )
    compare_parser.add_argument(
        "--output-dir",
        default="runs",
        help="directory containing validation runs",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        artifacts = run_experiment(
            args.config,
            output_dir=args.output_dir,
            query_limit=args.query_limit,
        )
        print(f"Run directory: {artifacts.run_directory}")
        print(f"Submission: {artifacts.submission_path}")
    elif args.command == "evaluate":
        artifacts = evaluate_experiment(
            args.config,
            output_dir=args.output_dir,
            query_limit=args.query_limit,
        )
        print(f"Run directory: {artifacts.run_directory}")
        print(f"Metrics: {artifacts.metrics_path}")
    elif args.command == "compare":
        artifacts = compare_evaluations(args.output_dir)
        print(artifacts.summary)
        print(f"Summary: {artifacts.summary_path}")
