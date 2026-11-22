import csv
import shutil

import pytest

from team_b.src.evaluation import RESULTS_CSV, write_experiment


@pytest.fixture
def log(tmp_path):
    """A temporary copy of the real results.csv, so tests never touch it."""
    copy = tmp_path / "results.csv"
    shutil.copy(RESULTS_CSV, copy)
    return copy


def test_appends_row_and_computes_win_rate(log):
    write_experiment(
        {"member": "Sadaf", "experiment_id": "T1", "model_id": "B-SA1",
         "status": "smoke", "matches": 10, "wins": 6, "losses": 4, "draws": 0},
        path=log,
    )
    rows = list(csv.DictReader(open(log)))
    assert len(rows) == 1
    assert rows[0]["team"] == "B"
    assert rows[0]["win_rate"] == "0.6"


def test_rejects_typo_column(log):
    with pytest.raises(ValueError, match="Unknown column"):
        write_experiment({"member": "Sadaf", "experiment_id": "T2", "model_id": "B-SA1",
                          "status": "smoke", "winrate": 0.5}, path=log)


def test_rejects_missing_required(log):
    with pytest.raises(ValueError, match="Missing required"):
        write_experiment({"member": "Sadaf"}, path=log)



from team_b.src.evaluation import summarize_matches


def test_summarize_matches_counts_and_win_rate():
    s = summarize_matches(["win", "win", "loss", "draw", "win", "error"])
    assert (s["wins"], s["losses"], s["draws"], s["crashes"]) == (3, 1, 1, 1)
    assert s["matches"] == 5          # errors are not counted as completed matches
    assert s["win_rate"] == 0.6
    assert 0 <= s["interval_lower"] < 0.6 < s["interval_upper"] <= 1


def test_summarize_matches_empty():
    s = summarize_matches([])
    assert s["matches"] == 0 and s["win_rate"] == 0.0
