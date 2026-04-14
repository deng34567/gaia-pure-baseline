from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from gaia_dataset import load_dataset
from gaia_scoring import check_gaia, extract_final_answer
from openai_client import OpenAIClientConfig, OpenAICompatibleClient
from prompts import build_messages
from reporting import (
    build_metrics_by_attachment_presence,
    build_metrics_by_level,
    build_summary,
    ensure_dir,
    write_json,
    write_jsonl,
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the GAIA pure single-model weak baseline.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--gaia-data-dir", default=None, help="Override GAIA dataset directory.")
    parser.add_argument("--split", choices=["validation", "test"], default=None, help="Override dataset split.")
    parser.add_argument("--level", type=int, choices=[1, 2, 3], default=None, help="Override GAIA level.")
    parser.add_argument("--limit", type=int, default=None, help="Override number of examples to process.")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    return parser.parse_args(argv)


def load_yaml_config(path: str) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        yaml = None

    config_path = Path(path).expanduser().resolve()
    if yaml is not None:
        with config_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise ValueError("Config file must define a mapping.")
        return data

    data: Dict[str, Any] = {}
    with config_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                raise ValueError(
                    "PyYAML is not installed and config line {} is not supported by the fallback parser: {!r}".format(
                        line_number, raw_line.rstrip("\n")
                    )
                )
            key, value = line.split(":", 1)
            data[key.strip()] = value.strip()
    return data


def resolve_config(args: argparse.Namespace) -> Dict[str, Any]:
    config = load_yaml_config(args.config)
    overrides = {
        "gaia_data_dir": args.gaia_data_dir,
        "split": args.split,
        "level": args.level,
        "limit": args.limit,
        "output_dir": args.output_dir,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value

    required_keys = [
        "model_name",
        "base_url",
        "api_key",
        "timeout_seconds",
        "max_tokens",
        "temperature",
        "gaia_data_dir",
        "split",
        "level",
        "limit",
        "output_dir",
    ]
    missing = [key for key in required_keys if key not in config]
    if missing:
        raise ValueError("Missing required config keys: {}".format(", ".join(missing)))

    config["timeout_seconds"] = float(config["timeout_seconds"])
    config["max_tokens"] = int(config["max_tokens"])
    config["temperature"] = float(config["temperature"])
    config["level"] = int(config["level"])
    config["limit"] = None if config["limit"] in (None, "", 0) else int(config["limit"])
    return config


def build_client(config: Dict[str, Any]) -> OpenAICompatibleClient:
    client_config = OpenAIClientConfig(
        model_name=str(config["model_name"]),
        base_url=str(config["base_url"]),
        api_key=str(config["api_key"]),
        timeout_seconds=float(config["timeout_seconds"]),
        max_tokens=int(config["max_tokens"]),
        temperature=float(config["temperature"]),
    )
    return OpenAICompatibleClient(client_config)


def run_evaluation(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    tasks = load_dataset(
        split=config["split"],
        level=config["level"],
        data_limit=config["limit"],
        data_dir=config["gaia_data_dir"],
    )
    client = build_client(config)

    results: List[Dict[str, Any]] = []
    total = len(tasks)
    for index, task in enumerate(tasks, start=1):
        raw_response = ""
        pred = ""
        correct = None

        try:
            messages = build_messages(task["question"], task["has_attachment"])
            raw_response = client.generate(messages)
            pred = extract_final_answer(raw_response) or ""
            if task["answer"] not in (None, ""):
                correct = check_gaia(pred, task["answer"])
        except Exception as exc:
            raw_response = "ERROR: {}".format(exc)
            pred = ""
            if task["answer"] not in (None, ""):
                correct = False

        record = {
            "id": task["id"],
            "level": task["level"],
            "question": task["question"],
            "has_attachment": task["has_attachment"],
            "raw_response": raw_response,
            "pred": pred,
            "answer": task["answer"],
            "correct": correct,
        }
        results.append(record)

        print(
            "[{}/{}] id={} level={} attachment={} correct={} pred={}".format(
                index,
                total,
                task["id"],
                task["level"],
                task["has_attachment"],
                correct,
                repr(pred),
            ),
            flush=True,
        )

    return results


def save_outputs(results: List[Dict[str, Any]], config: Dict[str, Any]) -> Path:
    output_dir = ensure_dir(config["output_dir"])
    write_jsonl(output_dir / "predictions.jsonl", results)
    write_json(output_dir / "summary.json", build_summary(results, config))
    write_json(output_dir / "metrics_by_level.json", build_metrics_by_level(results))
    write_json(
        output_dir / "metrics_by_attachment_presence.json",
        build_metrics_by_attachment_presence(results),
    )
    return output_dir


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    config = resolve_config(args)
    results = run_evaluation(config)
    output_dir = save_outputs(results, config)
    summary = build_summary(results, config)

    if summary["accuracy"] is None:
        print(
            "Finished {} examples from {} split. No local accuracy was computed.".format(
                summary["count"], config["split"]
            )
        )
    else:
        print(
            "Finished {} examples. Accuracy: {}/{} = {:.4f}".format(
                summary["count"],
                summary["correct_count"],
                summary["scored_count"],
                summary["accuracy"],
            )
        )
    print("Outputs written to {}".format(output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
