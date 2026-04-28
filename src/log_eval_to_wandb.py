#!/usr/bin/env python3
"""Log eval metrics and artifacts to Weights & Biases."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def flatten_metrics(prefix: str, value: Any, output: dict[str, float]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            flatten_metrics(f"{prefix}/{key}", child, output)
        return

    if isinstance(value, bool):
        output[prefix] = float(value)
        return

    if isinstance(value, (int, float)):
        output[prefix] = value


def sanitize_artifact_name(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return sanitized or "eval-artifact"


def add_summary_path(summary: Any, key: str, raw_path: str) -> None:
    if not raw_path:
        return
    path = Path(raw_path)
    if path.exists():
        summary[key] = str(path.resolve())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Log eval results to wandb.")
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--project", default="")
    parser.add_argument("--group", default="")
    parser.add_argument("--job-type", default="eval")
    parser.add_argument("--stage-label", default="eval")
    parser.add_argument("--run-time", required=True)
    parser.add_argument("--dataset-mode", default="dev")
    parser.add_argument("--prompt-name", default="direct")
    parser.add_argument("--eval-mode", default="major_voting")
    parser.add_argument("--eval-step", default="sql_generate")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--metric-json", default="")
    parser.add_argument("--predicted-sql", default="")
    parser.add_argument("--raw-pred-json", default="")
    parser.add_argument("--arg-json", default="")
    parser.add_argument("--wrapper-log", default="")
    parser.add_argument("--pipeline-log", default="")
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--artifact", action="append", default=[])
    parser.add_argument("--exit-code", type=int, default=0)
    parser.add_argument("--strict", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.report_to != "wandb":
        print("Skip eval wandb logging because report_to != wandb.", file=sys.stderr)
        return 0

    try:
        import wandb
    except Exception as exc:
        message = f"Failed to import wandb for eval logging: {exc}"
        if args.strict:
            raise RuntimeError(message) from exc
        print(message, file=sys.stderr)
        return 0

    config = {
        "stage_label": args.stage_label,
        "run_time": args.run_time,
        "dataset_mode": args.dataset_mode,
        "prompt_name": args.prompt_name,
        "eval_mode": args.eval_mode,
        "eval_step": args.eval_step,
        "model_path": args.model_path,
        "output_dir": args.output_dir,
    }

    run = None
    try:
        run = wandb.init(
            project=args.project or None,
            group=args.group or None,
            name=args.run_name,
            job_type=args.job_type,
            tags=args.tag or None,
            config=config,
            dir=os.getenv("WANDB_DIR"),
            reinit=True,
        )
        if run is None:
            raise RuntimeError("wandb.init returned None")

        metrics_to_log: dict[str, float] = {}
        metric_path = Path(args.metric_json) if args.metric_json else None
        metric_payload: dict[str, Any] | None = None
        if metric_path and metric_path.exists():
            raw_metric_payload = load_json(metric_path)
            if isinstance(raw_metric_payload, dict):
                metric_payload = raw_metric_payload
                if isinstance(raw_metric_payload.get("mj_metric"), dict):
                    flatten_metrics("eval/mj", raw_metric_payload["mj_metric"], metrics_to_log)
                if isinstance(raw_metric_payload.get("pass_at_k_metric"), dict):
                    flatten_metrics("eval/pass_at_k", raw_metric_payload["pass_at_k_metric"], metrics_to_log)
                if isinstance(raw_metric_payload.get("mj_pass_at_2_metric"), dict):
                    flatten_metrics("eval/pass_at_2", raw_metric_payload["mj_pass_at_2_metric"], metrics_to_log)
                if not metrics_to_log:
                    flatten_metrics("eval", raw_metric_payload, metrics_to_log)

        if metrics_to_log:
            wandb.log(metrics_to_log)

        summary = run.summary
        summary["status"] = "success" if args.exit_code == 0 else "failed"
        summary["exit_code"] = args.exit_code
        summary["stage_label"] = args.stage_label
        summary["run_time"] = args.run_time
        summary["dataset_mode"] = args.dataset_mode
        summary["prompt_name"] = args.prompt_name
        summary["eval_mode"] = args.eval_mode
        summary["eval_step"] = args.eval_step
        if args.model_path:
            summary["model_path"] = args.model_path

        add_summary_path(summary, "output_dir", args.output_dir)
        add_summary_path(summary, "metric_json_path", args.metric_json)
        add_summary_path(summary, "predicted_sql_path", args.predicted_sql)
        add_summary_path(summary, "raw_pred_json_path", args.raw_pred_json)
        add_summary_path(summary, "arg_json_path", args.arg_json)
        add_summary_path(summary, "wrapper_log_path", args.wrapper_log)
        add_summary_path(summary, "pipeline_log_path", args.pipeline_log)

        if metric_payload:
            for prefix, key in (
                ("eval_mj", "mj_metric"),
                ("eval_pass_at_k", "pass_at_k_metric"),
                ("eval_pass_at_2", "mj_pass_at_2_metric"),
            ):
                value = metric_payload.get(key)
                if isinstance(value, dict):
                    for sub_key in ("all", "acc", "easy", "medium", "hard", "all_total"):
                        if sub_key in value and isinstance(value[sub_key], (int, float)):
                            summary[f"{prefix}_{sub_key}"] = value[sub_key]

        artifact_paths: list[Path] = []
        for raw_path in (
            args.metric_json,
            args.predicted_sql,
            args.raw_pred_json,
            args.arg_json,
            args.wrapper_log,
            args.pipeline_log,
            *args.artifact,
        ):
            if not raw_path:
                continue
            path = Path(raw_path)
            if path.exists():
                artifact_paths.append(path)

        if artifact_paths:
            artifact = wandb.Artifact(
                name=sanitize_artifact_name(f"{args.run_name}-{args.run_time}"),
                type="eval-results",
            )
            for path in artifact_paths:
                artifact.add_file(str(path), name=path.name)
            run.log_artifact(artifact)

        wandb.finish(exit_code=args.exit_code)
        return 0
    except Exception as exc:
        if run is not None:
            try:
                wandb.finish(exit_code=args.exit_code or 1)
            except Exception:
                pass

        message = f"Failed to log eval run to wandb: {exc}"
        if args.strict:
            raise RuntimeError(message) from exc
        print(message, file=sys.stderr)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
