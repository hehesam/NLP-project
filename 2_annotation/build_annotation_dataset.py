import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


EXPECTED_MODELS = ("ge", "cl", "gpt")
FILE_PATTERN = re.compile(r"^batch(\d+)_([a-zA-Z0-9]+)\.jsonl$")


def _to_float(value: Any) -> Optional[float]:
	if value is None or value == "":
		return None
	try:
		return float(value)
	except (TypeError, ValueError):
		return None


def normalize_record(raw: Dict[str, Any], source: Path) -> Optional[Dict[str, Any]]:
	email_id = raw.get("id")
	if email_id is None:
		return None

	return {
		"id": str(email_id).strip(),
		"topic_label": str(raw.get("topic_label", "")).strip() or None,
		"priority_label": str(raw.get("priority_label", "")).strip() or None,
		"topic_confidence": _to_float(raw.get("topic_confidence")),
		"priority_confidence": _to_float(raw.get("priority_confidence")),
		"short_rationale": str(raw.get("short_rationale", "")).strip() or None,
		"_source_file": source.name,
	}


def parse_annotation_file(path: Path) -> List[Dict[str, Any]]:
	text = path.read_text(encoding="utf-8", errors="replace").lstrip("\ufeff")

	records: List[Dict[str, Any]] = []

	# First, try standard JSON (list or single object).
	try:
		parsed = json.loads(text)
		if isinstance(parsed, dict):
			parsed = [parsed]
		if isinstance(parsed, list):
			for item in parsed:
				if isinstance(item, dict):
					rec = normalize_record(item, path)
					if rec:
						records.append(rec)
			if records:
				return records
	except json.JSONDecodeError:
		pass

	# Fallback for noisy files: parse line-by-line JSON objects,
	# skipping wrappers such as markdown fences and assistant preambles.
	for raw_line in text.splitlines():
		line = raw_line.strip()
		if not line:
			continue
		if line.startswith("```"):
			continue
		if line.lower().startswith("read "):
			continue
		if line.lower().startswith("i'll read"):
			continue
		if not (line.startswith("{") and line.endswith("}")):
			continue

		try:
			obj = json.loads(line)
		except json.JSONDecodeError:
			continue

		if isinstance(obj, dict):
			rec = normalize_record(obj, path)
			if rec:
				records.append(rec)

	return records


def find_batch_files(input_dir: Path) -> Dict[int, Dict[str, Path]]:
	grouped: Dict[int, Dict[str, Path]] = defaultdict(dict)
	for path in sorted(input_dir.glob("batch*_*.jsonl")):
		match = FILE_PATTERN.match(path.name)
		if not match:
			continue
		batch_num = int(match.group(1))
		model = match.group(2).lower()
		grouped[batch_num][model] = path
	return dict(grouped)


def weighted_consensus(values: Iterable[Tuple[Optional[str], Optional[float]]]) -> Tuple[Optional[str], int]:
	score_by_label: Dict[str, float] = defaultdict(float)
	count_by_label: Dict[str, int] = defaultdict(int)

	for label, confidence in values:
		if not label:
			continue
		weight = confidence if confidence is not None else 1.0
		score_by_label[label] += weight
		count_by_label[label] += 1

	if not score_by_label:
		return None, 0

	# Sort by weighted score, then by vote count, then label for deterministic output.
	best = sorted(
		score_by_label.keys(),
		key=lambda label: (score_by_label[label], count_by_label[label], label),
		reverse=True,
	)[0]
	return best, count_by_label[best]


def merge_batches(input_dir: Path, models: Tuple[str, ...]) -> Tuple[List[Dict[str, Any]], List[str]]:
	batch_files = find_batch_files(input_dir)
	if not batch_files:
		raise FileNotFoundError(f"No batch files found in: {input_dir}")

	merged_rows: List[Dict[str, Any]] = []
	warnings: List[str] = []

	for batch_num in sorted(batch_files):
		model_to_path = batch_files[batch_num]

		records_by_model: Dict[str, Dict[str, Dict[str, Any]]] = {}
		for model in models:
			file_path = model_to_path.get(model)
			if not file_path:
				warnings.append(f"batch {batch_num:02d}: missing model file for '{model}'")
				records_by_model[model] = {}
				continue

			parsed_records = parse_annotation_file(file_path)
			by_id: Dict[str, Dict[str, Any]] = {}
			for rec in parsed_records:
				rid = rec["id"]
				if rid in by_id:
					warnings.append(
						f"batch {batch_num:02d}, model {model}: duplicate id '{rid}' (keeping last occurrence)"
					)
				by_id[rid] = rec
			records_by_model[model] = by_id

		all_ids = sorted({rid for by_id in records_by_model.values() for rid in by_id})

		for rid in all_ids:
			row: Dict[str, Any] = {
				"batch": batch_num,
				"id": rid,
			}

			topic_votes: List[Tuple[Optional[str], Optional[float]]] = []
			priority_votes: List[Tuple[Optional[str], Optional[float]]] = []
			present_models: List[str] = []

			for model in models:
				rec = records_by_model[model].get(rid)
				topic_label = rec.get("topic_label") if rec else None
				priority_label = rec.get("priority_label") if rec else None
				topic_conf = rec.get("topic_confidence") if rec else None
				priority_conf = rec.get("priority_confidence") if rec else None
				rationale = rec.get("short_rationale") if rec else None

				if rec:
					present_models.append(model)

				row[f"{model}_topic_label"] = topic_label
				row[f"{model}_topic_confidence"] = topic_conf
				row[f"{model}_priority_label"] = priority_label
				row[f"{model}_priority_confidence"] = priority_conf
				row[f"{model}_short_rationale"] = rationale

				topic_votes.append((topic_label, topic_conf))
				priority_votes.append((priority_label, priority_conf))

			topic_consensus, topic_agreement = weighted_consensus(topic_votes)
			priority_consensus, priority_agreement = weighted_consensus(priority_votes)

			row["topic_consensus_label"] = topic_consensus
			row["topic_agreement_count"] = topic_agreement
			row["priority_consensus_label"] = priority_consensus
			row["priority_agreement_count"] = priority_agreement
			row["available_models"] = ",".join(present_models)

			if len(present_models) != len(models):
				missing = ",".join([m for m in models if m not in present_models])
				warnings.append(f"batch {batch_num:02d}, id {rid}: missing annotations from {missing}")

			merged_rows.append(row)

	return merged_rows, warnings


def write_jsonl(rows: List[Dict[str, Any]], output_path: Path) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", encoding="utf-8", newline="\n") as f:
		for row in rows:
			f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
	if not rows:
		return
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fieldnames = list(rows[0].keys())
	with output_path.open("w", encoding="utf-8", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Merge GE/CL/GPT annotation batches into a single consolidated dataset."
	)
	parser.add_argument(
		"--input-dir",
		default="../data/labels",
		help="Directory containing files like batch01_ge.jsonl, batch01_cl.jsonl, batch01_gpt.jsonl",
	)
	parser.add_argument(
		"--output-jsonl",
		default="../data/annotation/annotation_dataset.jsonl",
		help="Path to write the merged JSONL dataset",
	)
	parser.add_argument(
		"--output-csv",
		default="../data/annotation/annotation_dataset.csv",
		help="Path to write the merged CSV dataset",
	)
	parser.add_argument(
		"--models",
		default=",".join(EXPECTED_MODELS),
		help="Comma-separated model suffixes to merge (default: ge,cl,gpt)",
	)
	parser.add_argument(
		"--quiet",
		action="store_true",
		help="Suppress warning output",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	script_dir = Path(__file__).resolve().parent
	input_dir = (script_dir / args.input_dir).resolve() if not Path(args.input_dir).is_absolute() else Path(args.input_dir)
	output_jsonl = (
		(script_dir / args.output_jsonl).resolve()
		if not Path(args.output_jsonl).is_absolute()
		else Path(args.output_jsonl)
	)
	output_csv = (
		(script_dir / args.output_csv).resolve()
		if not Path(args.output_csv).is_absolute()
		else Path(args.output_csv)
	)

	models = tuple(m.strip().lower() for m in args.models.split(",") if m.strip())
	if not models:
		raise ValueError("No valid models provided via --models")

	rows, warnings = merge_batches(input_dir=input_dir, models=models)

	rows = sorted(rows, key=lambda r: (r["batch"], r["id"]))
	write_jsonl(rows, output_jsonl)
	write_csv(rows, output_csv)

	print(f"Merged rows: {len(rows)}")
	print(f"Output JSONL: {output_jsonl}")
	print(f"Output CSV: {output_csv}")
	print(f"Models: {', '.join(models)}")

	if warnings and not args.quiet:
		print(f"Warnings ({len(warnings)}):")
		for w in warnings:
			print(f"- {w}")


if __name__ == "__main__":
	main()
