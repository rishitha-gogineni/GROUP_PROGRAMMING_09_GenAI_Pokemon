"""Tests for team_b/src/evaluation.py (standard-library unittest only).

Run: python3 -m unittest discover -s team_b/tests -t . -v
"""

import csv
import shutil
import tempfile
import unittest
from pathlib import Path

from team_b.src.evaluation import (
    RESULTS_CSV,
    outcomes_from_series,
    summarize_matches,
    write_experiment,
)


class WriteExperimentTests(unittest.TestCase):
    def setUp(self):
        # A temporary copy of the real results.csv, so tests never touch it.
        self._tmp = tempfile.TemporaryDirectory()
        self.log = Path(self._tmp.name) / "results.csv"
        shutil.copy(RESULTS_CSV, self.log)

    def tearDown(self):
        self._tmp.cleanup()

    def rows(self):
        with open(self.log, newline="") as f:
            return list(csv.DictReader(f))

    def test_appends_row_and_computes_win_rate(self):
        before = len(self.rows())  # log may already hold real rows
        write_experiment({"member": "Sadaf", "experiment_id": "T1", "model_id": "B-SA1",
                          "status": "smoke", "matches": 10, "wins": 6, "losses": 4, "draws": 0},
                         path=self.log)
        rows = self.rows()
        self.assertEqual(len(rows), before + 1)
        self.assertEqual(rows[-1]["team"], "B")
        self.assertEqual(rows[-1]["win_rate"], "0.6")

    def test_uses_unix_line_endings(self):
        write_experiment({"member": "Sadaf", "experiment_id": "T2", "model_id": "B-SA1",
                          "status": "smoke"}, path=self.log)
        self.assertNotIn(b"\r\n", self.log.read_bytes())

    def test_text_zero_matches_does_not_crash(self):
        row = write_experiment({"member": "Sadaf", "experiment_id": "T3", "model_id": "B-SA1",
                                "status": "smoke", "matches": "0", "wins": "0"}, path=self.log)
        self.assertEqual(row["win_rate"], "")

    def test_rejects_typo_column(self):
        with self.assertRaisesRegex(ValueError, "Unknown column"):
            write_experiment({"member": "Sadaf", "experiment_id": "T4", "model_id": "B-SA1",
                              "status": "smoke", "winrate": 0.5}, path=self.log)

    def test_rejects_missing_required(self):
        with self.assertRaisesRegex(ValueError, "Missing required"):
            write_experiment({"member": "Sadaf"}, path=self.log)


class SummarizeMatchesTests(unittest.TestCase):
    def test_counts_and_win_rate(self):
        s = summarize_matches(["win", "win", "loss", "draw", "win", "error"])
        self.assertEqual((s["wins"], s["losses"], s["draws"], s["crashes"]), (3, 1, 1, 1))
        self.assertEqual(s["matches"], 5)  # the error is not a completed match
        self.assertEqual(s["win_rate"], 0.6)
        self.assertEqual(s["interval_method"], "wilson95")
        self.assertTrue(0 <= s["interval_lower"] < 0.6 < s["interval_upper"] <= 1)

    def test_empty(self):
        s = summarize_matches([])
        self.assertEqual((s["matches"], s["win_rate"]), (0, 0.0))

    def test_aborted_series_excluded_from_win_rate(self):
        # Chelsi's simulator reports an aborted best-of-three series as "E".
        summary = {"match_results": [{"result_for_a": r} for r in ("W", "E", "L", "W", "D")]}
        outcomes = outcomes_from_series(summary)
        self.assertEqual(outcomes, ["win", "error", "loss", "win", "draw"])
        s = summarize_matches(outcomes)
        self.assertEqual((s["matches"], s["crashes"]), (4, 1))
        self.assertEqual(s["win_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
