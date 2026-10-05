import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def load_config(path="configs/config.yaml"):
    with open(ROOT / path) as f:
        return yaml.safe_load(f)

CFG = load_config()