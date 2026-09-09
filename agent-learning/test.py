from pathlib import Path
csv_path = Path(__file__).resolve().parent.parent/"students.csv"
lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
print(len(lines))