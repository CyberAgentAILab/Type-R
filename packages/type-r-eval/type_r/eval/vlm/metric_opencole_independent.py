import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = [
    "design_and_layout",
    "content_relevance_and_effectiveness",
    "typography_and_color_scheme",
    "graphics_and_images",
    "innovation_and_originality",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir", type=str, required=True)
    args = parser.parse_args()

    data = []
    for key in KEYS:
        df = pd.read_csv(str(Path(args.csv_dir) / f"{key}.csv"), dtype={"id": object})
        print(f"{key}: {df['score'].mean():.2f} ± {df['score'].std():.2f}")
        data.extend(df["score"].values)

    data = np.array(data)
    print(f"Total: {data.mean():.2f} ± {data.std():.2f}")
