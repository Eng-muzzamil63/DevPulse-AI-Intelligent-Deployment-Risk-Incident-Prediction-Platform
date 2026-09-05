from pathlib import Path
r=Path(__file__).resolve().parents[1]
assert (r/'backend/app/main.py').exists()
assert (r/'frontend/src/App.jsx').exists()
assert (r/'ml/train.py').exists()
