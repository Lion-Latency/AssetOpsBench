from __future__ import annotations
import argparse, json, csv, datetime, traceback
from pathlib import Path
from typing import List, Dict, Any
from pydantic import ValidationError
from scenarios_schema import load_scenario, load_utterance


def try_load_json(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        objs = []
        for i, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                objs.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSONL parse error in {path.name} line {i}: {e}")
        return objs
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON in {path.name}: {e}")
    return parsed if isinstance(parsed, list) else [parsed]


def validate_objects(objs: List[Dict[str, Any]]):
    results = []
    for idx, obj in enumerate(objs):
        item = {
            "index": idx,
            "is_scenario_like": any(k in obj for k in ("uuid", "execution", "environment")),
            "utterance_ok": False,
            "scenario_ok": False,
            "errors": []
        }
        try:
            load_utterance(obj)
            item["utterance_ok"] = True
        except ValidationError as e:
            item["errors"].append({"kind": "utterance", "error": e.errors()})
        except Exception as e:
            item["errors"].append({"kind": "utterance", "error": str(e)})

        if item["is_scenario_like"]:
            try:
                load_scenario(obj)
                item["scenario_ok"] = True
            except ValidationError as e:
                item["errors"].append({"kind": "scenario", "error": e.errors()})
            except Exception as e:
                item["errors"].append({"kind": "scenario", "error": str(e)})
        results.append(item)
    return results


def run_validation(input_path: Path, out_dir: Path):
    input_path = input_path.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        files = [input_path] if input_path.suffix.lower() in (".json", ".jsonl") else []
    elif input_path.is_dir():
        files = sorted(p for p in input_path.iterdir() if p.suffix.lower() in (".json", ".jsonl"))
    else:
        raise FileNotFoundError(f"Input not found: {input_path}")

    report = {
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "input": str(input_path),
        "files": {}
    }
    summary_rows = []

    for f in files:
        entry = {
            "path": str(f),
            "parsed": False,
            "num_objects": 0,
            "utterance_valid": 0,
            "utterance_invalid": 0,
            "scenario_valid": 0,
            "scenario_invalid": 0,
            "errors": []
        }
        try:
            objs = try_load_json(f)
            entry["parsed"] = True
            entry["num_objects"] = len(objs)
            results = validate_objects(objs)
            for r in results:
                if r["utterance_ok"]:
                    entry["utterance_valid"] += 1
                else:
                    entry["utterance_invalid"] += 1
                    entry["errors"].extend(r["errors"])
                if r["is_scenario_like"]:
                    if r["scenario_ok"]:
                        entry["scenario_valid"] += 1
                    else:
                        entry["scenario_invalid"] += 1
                        entry["errors"].extend(r["errors"])
        except Exception as e:
            entry["parse_error"] = str(e)
            entry["traceback"] = traceback.format_exc()
        report["files"][f.name] = entry
        summary_rows.append({
            "file": f.name,
            "parsed": entry["parsed"],
            "num_objects": entry["num_objects"],
            "utterance_valid": entry["utterance_valid"],
            "utterance_invalid": entry["utterance_invalid"],
            "scenario_valid": entry["scenario_valid"],
            "scenario_invalid": entry["scenario_invalid"],
        })

    out_json = out_dir / "validation_report.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    out_csv = out_dir / "validation_summary.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=summary_rows[0].keys() if summary_rows else [])
        if summary_rows:
            writer.writeheader()
            writer.writerows(summary_rows)

    summary_txt = out_dir / "summary.txt"
    with summary_txt.open("w", encoding="utf-8") as fh:
        fh.write(f"Validation run: {report['generated_at']}\nInput: {input_path}\nFiles: {len(files)}\n\n")
        for fname, info in report["files"].items():
            fh.write(f"- {fname}: utt_ok={info.get('utterance_valid')} "
                     f"utt_err={info.get('utterance_invalid')} "
                     f"scen_ok={info.get('scenario_valid')} "
                     f"scen_err={info.get('scenario_invalid')}\n")

    print("Validation complete. Results in:", out_dir)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate .json/.jsonl scenario & utterance files")
    parser.add_argument("input", help="File or directory to validate")
    parser.add_argument("--out-dir", "-o", default="./validation_output", help="Output directory")
    args = parser.parse_args()
    run_validation(Path(args.input), Path(args.out_dir))
