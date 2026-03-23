import pandas as pd
import json
import random
import argparse
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple
from sklearn.model_selection import train_test_split

@dataclass
class AlpacaSample:
    instruction: str
    input: str
    output: str
    isBeOrAf: str

class XLSXProcessor:
    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.instruction_text = "Detect whether the following code contains vulnerabilities."

    def extract_from_xlsx(self):
        """Extracts data from sheet 'CodeQL' and 'Traditional' from all XLSX files."""
        codeql_samples: List[AlpacaSample] = []
        traditional_samples: List[AlpacaSample] = []

        xlsx_files = list(self.data_dir.glob("*.xlsx"))
        print(f"[EXTRACT] Found {len(xlsx_files)} .xlsx files.")

        for file_path in tqdm(xlsx_files, desc="Processing files"):
            if file_path.name.startswith(".~"): continue
            try:
                all_sheets = pd.read_excel(file_path, sheet_name=None)
                if "CodeQL" in all_sheets:
                    codeql_samples.extend(self._extract_samples(all_sheets["CodeQL"]))
                if "Traditional" in all_sheets:
                    traditional_samples.extend(self._extract_samples(all_sheets["Traditional"]))
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

        # Save aggregated JSONs
        self._save_json(codeql_samples, "CodeQL.json")
        self._save_json(traditional_samples, "Traditional.json")
        
        print(f"[EXTRACT] Saved {len(codeql_samples)} samples to CodeQL.json")
        print(f"[EXTRACT] Saved {len(traditional_samples)} samples to Traditional.json")

    def split_json(self, json_name: str, ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1)):
        """Splits a JSON file into train, val, and test. Maintains vul/safe ratio."""
        json_path = self.output_dir / json_name
        if not json_path.exists():
            print(f"File {json_path} does not exist. Run extractor first.")
            return

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not data:
            print(f"File {json_name} is empty.")
            return

        print(f"\n--- SPLITTING DATASET: {json_name} ---")
        self._print_stats(data, "Original")

        # Stratified Split logic (8:1:1)
        # Split 1: train (80%) vs temporary (20%)
        # Split 2: temporary (20%) -> validation (10% total) vs test (10% total)
        train_data, temp_data = train_test_split(
            data,
            test_size=(ratio[1] + ratio[2]),
            stratify=[s["output"] for s in data],
            random_state=42,
            shuffle=True
        )
        
        # Split the temporary 20% into two equal 10% halves
        val_data, test_data = train_test_split(
            temp_data,
            test_size=ratio[2] / (ratio[1] + ratio[2]),
            stratify=[s["output"] for s in temp_data],
            random_state=42,
            shuffle=True
        )

        # Output results to separate folder
        folder_name = json_name.replace(".json", "")
        split_dir = self.output_dir / folder_name
        split_dir.mkdir(parents=True, exist_ok=True)

        self._save_json_raw(train_data, split_dir / "train.json")
        self._save_json_raw(val_data, split_dir / "val.json")
        self._save_json_raw(test_data, split_dir / "test.json")

        self._print_stats(train_data, "Train (80%)")
        self._print_stats(val_data, "Val (10%)")
        self._print_stats(test_data, "Test (10%)")
        print(f"Split completed for {json_name}. Folder: {split_dir}")

    def _extract_samples(self, df: pd.DataFrame) -> List[AlpacaSample]:
        samples = []
        required_cols = ["function_code", "is_vul", "isBeOrAf"]
        for col in required_cols:
            if col not in df.columns: return []

        for _, row in df.iterrows():
            input_code = str(row["function_code"]) if pd.notna(row["function_code"]) else ""
            raw_output = str(row["is_vul"]).strip() if pd.notna(row["is_vul"]) else "0"

            # Handle type float 1.0 to int 1
            try:
                output_val = str(int(float(raw_output)))
            except (ValueError, TypeError):
                output_val = raw_output

            is_be_or_af = str(row["isBeOrAf"]) if pd.notna(row["isBeOrAf"]) else ""

            samples.append(AlpacaSample(
                instruction=self.instruction_text,
                input=input_code,
                output=output_val,
                isBeOrAf=is_be_or_af
            ))
        return samples

    def _print_stats(self, data: List[Dict[str, Any]], title: str):
        total = len(data)
        if total == 0:
            print(f"{title}: Empty list.")
            return
        vuls = sum(1 for s in data if s["output"] == "1")
        safes = total - vuls
        vul_ratio = (vuls / total) * 100
        safe_ratio = (safes / total) * 100
        print(f"[{title}] Total: {total} | Vul: {vuls} ({vul_ratio:.2f}%) | Safe: {safes} ({safe_ratio:.2f}%)")

    def _save_json(self, samples: List[AlpacaSample], filename: str):
        output_path = self.output_dir / filename
        data = [asdict(s) for s in samples]
        self._save_json_raw(data, output_path)

    def _save_json_raw(self, data: List[Dict[str, Any]], output_path: Path):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser(description="Process and Split XLSX data for LLM Fine-tuning")
    parser.add_argument("mode", choices=["extractor", "split", "all"], help="Choose operation mode")
    parser.add_argument("--data_dir", default="data/SamplesFiles", help="Path to raw XLSX folder")
    parser.add_argument("--output_dir", default="data/processed", help="Path to output folder")
    
    args = parser.parse_args()
    processor = XLSXProcessor(args.data_dir, args.output_dir)

    if args.mode == "extractor" or args.mode == "all":
        processor.extract_from_xlsx()

    if args.mode == "split" or args.mode == "all":
        # Split both generated files if they exist
        processor.split_json("CodeQL.json")
        processor.split_json("Traditional.json")

if __name__ == "__main__":
    main()
