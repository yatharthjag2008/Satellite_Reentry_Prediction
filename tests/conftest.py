import sys
from pathlib import Path

# The scripts in src/ import each other by module name, so make them importable here too.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
