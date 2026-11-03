from pathlib import Path

from utis.targets import STJ_LOWER

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PAPER_CHECKPOINT = ROOT / "trained_models" / "UNet3plus.ckpt"
SCRIPTED_MODEL = ROOT / "trained_models" / "utis_model_scripted.pt"
REAL_SAMPLES = DATA / "real_samples" / STJ_LOWER
