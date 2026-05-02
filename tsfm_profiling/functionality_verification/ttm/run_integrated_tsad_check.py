from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

BASE_DIR = Path(__file__).resolve().parent
FUNCTIONALITY_VERIFICATION_DIR = BASE_DIR.parent
DATA_DIR = FUNCTIONALITY_VERIFICATION_DIR / "synthetic_data"
MODELS_DIR = repo_root / "src" / "servers" / "tsfm" / "artifacts" / "tsfm_models"

from src.servers.tsfm.main import run_integrated_tsad

tsad_dataset_path = str(DATA_DIR / "chiller9_tsad.csv")

print("\n" + "="*60)
print("INTEGRATED ANOMALY DETECTION CHECK")
print("="*60)

result = run_integrated_tsad(
    dataset_path=tsad_dataset_path,
    timestamp_column="Timestamp",
    target_columns=["Chiller 9 Condenser Water Flow"],
    model_checkpoint="ttm_96_28",
    false_alarm=0.05,
    n_calibration=0.2,
)

status = getattr(result, 'status', None)
if status == 'success':
    print("\n✓ STATUS: SUCCESS")
    message = getattr(result, 'message', '')
    if message:
        print(f"  Message: {message}")
    if hasattr(result, 'anomaly_count'):
        print(f"  Anomalies detected: {result.anomaly_count}/{result.total_records}")
    if hasattr(result, 'results_file'):
        print(f"  Results: {result.results_file}")
else:
    print("\n✗ STATUS: FAILED")
    error = getattr(result, 'error', str(result))
    print(f"  Error: {error}")
print("="*60)