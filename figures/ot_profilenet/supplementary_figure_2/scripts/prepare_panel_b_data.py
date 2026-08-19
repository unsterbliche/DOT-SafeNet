"""Convert V100-derived per-target counts into the panel-b source table."""
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "panel_b_counts_raw.txt"
OUTPUT = ROOT / "data" / "panel_b_compounds_per_target.csv"
def main():
    rows = []
    for line in RAW.read_text(encoding="utf-8").splitlines():
        count, target_id = line.split()
        rows.append({"target_id": target_id, "compound_count": int(count)})
    data = pd.DataFrame(rows).sort_values("target_id").reset_index(drop=True)
    if len(data) != 7258 or int(data["compound_count"].sum()) != 2102767:
        raise ValueError("Panel-b counts do not match the reported pretraining totals")
    data.to_csv(OUTPUT, index=False)
    print(f"wrote {OUTPUT}: {len(data)} targets, {data['compound_count'].sum()} pairs")
if __name__ == "__main__":
    main()