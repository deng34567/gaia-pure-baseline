import json
import unittest

from gaia_dataset import load_dataset, resolve_attachment_path


class GaiaDatasetTests(unittest.TestCase):
    def test_load_dataset_jsonl_fallback(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp_dir:
            from pathlib import Path

            tmp_path = Path(tmp_dir)
            split_dir = tmp_path / "2023" / "validation"
            split_dir.mkdir(parents=True)
            metadata_path = split_dir / "metadata.jsonl"

            rows = [
                {
                    "task_id": "task-1",
                    "Question": "Question one",
                    "Level": 1,
                    "Final answer": "alpha",
                    "file_name": None,
                    "file_path": None,
                },
                {
                    "task_id": "task-2",
                    "Question": "Question two",
                    "Level": 2,
                    "Final answer": "beta",
                    "file_name": "example.txt",
                    "file_path": None,
                },
            ]

            with metadata_path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

            tasks = load_dataset(split="validation", level=1, data_limit=None, data_dir=str(tmp_path))

            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0]["id"], "task-1")
            self.assertFalse(tasks[0]["has_attachment"])

    def test_resolve_attachment_path_prefers_file_path(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp_dir:
            from pathlib import Path

            tmp_path = Path(tmp_dir)
            row = {"file_path": "2023/validation/example.txt", "file_name": "ignored.txt"}
            path = resolve_attachment_path(row, data_dir=str(tmp_path), split="validation")
            self.assertEqual(path, str((tmp_path / "2023" / "validation" / "example.txt").resolve()))

    @unittest.skipUnless(__import__("importlib").util.find_spec("pyarrow") is not None, "pyarrow not installed")
    def test_load_dataset_parquet_level_split(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        import pyarrow
        import pyarrow.parquet as pq

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            split_dir = tmp_path / "2023" / "validation"
            split_dir.mkdir(parents=True)
            metadata_path = split_dir / "metadata.level2.parquet"

            rows = [
                {
                    "task_id": "task-2a",
                    "Question": "Question two",
                    "Level": 2,
                    "Final answer": "beta",
                    "file_name": None,
                    "file_path": None,
                },
                {
                    "task_id": "task-2b",
                    "Question": "Question three",
                    "Level": 2,
                    "Final answer": "gamma",
                    "file_name": "sheet.xlsx",
                    "file_path": None,
                },
            ]

            table = pyarrow.Table.from_pylist(rows)
            pq.write_table(table, metadata_path)

            tasks = load_dataset(split="validation", level=2, data_limit=None, data_dir=str(tmp_path))

            self.assertEqual([task["id"] for task in tasks], ["task-2a", "task-2b"])
            self.assertTrue(tasks[1]["has_attachment"])


if __name__ == "__main__":
    unittest.main()
