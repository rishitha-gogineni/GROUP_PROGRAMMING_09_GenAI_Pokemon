"""Team A shared packaging command. Run from any working directory."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from team_a.src.packaging import main


if __name__ == '__main__':
    raise SystemExit(main())
