from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "synthetic_data"
MODELS_DIR = repo_root / "src" / "servers" / "tsfm" / "artifacts" / "tsfm_models"

from src.servers.tsfm.main import run_tsfm_finetuning

print("\n" + "="*60)
print("FINE-TUNING CHECK")
print("="*60)

result = run_tsfm_finetuning(
    dataset_path=str(DATA_DIR / "chiller9_finetuning_small.csv"),
    timestamp_column="Timestamp",
    target_columns=["Chiller 9 Condenser Water Flow"],
    model_checkpoint="ttm_96_28",
    save_model_dir="tunedmodels",
    forecast_horizon=24,
    n_finetune=0.05,
    n_test=0.05,
)

status = getattr(result, 'status', None)
if status == 'success':
    print("\n✓ STATUS: SUCCESS")
    message = getattr(result, 'message', '')
    if message:
        print(f"  Message: {message}")
    if hasattr(result, 'model_checkpoint'):
        print(f"  Model: {result.model_checkpoint}")
    if hasattr(result, 'results_file'):
        print(f"  Results: {result.results_file}")
else:
    print("\n✗ STATUS: FAILED")
    error = getattr(result, 'error', str(result))
    print(f"  Error: {error}")
print("="*60)