import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from run_eval import main


class FakeClient:
    def __init__(self, answers):
        self._answers = list(answers)

    def generate(self, messages):
        return self._answers.pop(0)


class RunEvalSmokeTests(unittest.TestCase):
    def test_run_eval_writes_expected_outputs(self):
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            split_dir = tmp_path / "gaia" / "2023" / "validation"
            split_dir.mkdir(parents=True)
            metadata_path = split_dir / "metadata.jsonl"

            rows = [
                {
                    "task_id": "task-1",
                    "Question": "What is 2+2?",
                    "Level": 1,
                    "Final answer": "4",
                    "file_name": None,
                    "file_path": None,
                },
                {
                    "task_id": "task-2",
                    "Question": "State the word alpha.",
                    "Level": 1,
                    "Final answer": "alpha",
                    "file_name": "ignored.pdf",
                    "file_path": None,
                },
            ]

            with metadata_path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

            config_path = tmp_path / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "model_name: fake-model",
                        "base_url: http://127.0.0.1:8000/v1",
                        "api_key: sk-12345",
                        "timeout_seconds: 30",
                        "max_tokens: 256",
                        "temperature: 0.0",
                        f"gaia_data_dir: {tmp_path / 'gaia'}",
                        "split: validation",
                        "level: 1",
                        "limit: 2",
                        f"output_dir: {tmp_path / 'outputs'}",
                    ]
                ),
                encoding="utf-8",
            )

            with patch(
                "run_eval.build_client",
                lambda config: FakeClient(["FINAL ANSWER: 4", "FINAL ANSWER: alpha"]),
            ):
                exit_code = main(["--config", str(config_path)])

            self.assertEqual(exit_code, 0)

            output_dir = tmp_path / "outputs"
            predictions = (output_dir / "predictions.jsonl").read_text(encoding="utf-8").strip().splitlines()
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            metrics = json.loads(
                (output_dir / "metrics_by_attachment_presence.json").read_text(encoding="utf-8")
            )

            self.assertEqual(len(predictions), 2)
            self.assertEqual(summary["correct_count"], 2)
            self.assertEqual(summary["accuracy"], 1.0)
            self.assertEqual(metrics["with_attachment"]["count"], 1)
            self.assertEqual(metrics["without_attachment"]["count"], 1)


if __name__ == "__main__":
    unittest.main()
