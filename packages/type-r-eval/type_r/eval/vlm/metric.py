import argparse
from collections import defaultdict

import pandas as pd


def visualize_rating_results(df: pd.DataFrame, id_key: str) -> None:
    df = df.replace(-1, 1)  # -1 means failed to evaluate, thus give the lowest
    df = df.drop(columns=[id_key])
    for name, series in df.items():
        if name != "score":
            continue
        mean = series.mean().item()
        std = series.std().item()
        print(f"{name}: {mean:.2f} ± {std:.2f}")


# def dump_rating_table(dfs: dict[str, pd.DataFrame]) -> None:
#     # TODO: edit OpenCOLE's script and re-use it
#     text = ""
#     for model_name, df in dfs.items():
#         values = [df.loc[:, m].mean().item() for m in METRICS]
#         mean = sum(values) / len(values)
#         values = values + [str(mean)[:3]]
#         values_str = [str(v)[:3] for v in values]
#         text += f"\t{model_name} & " + " & ".join(values_str) + " \\\\\n"
#     print(text)


def visualize_voting_results(df: pd.DataFrame) -> None:
    N = len(df)

    # how many candidates to be compared (typically 2)
    C = sum([c.startswith("result:") for c in df.columns])

    count = defaultdict(float)
    for _, row in df.iterrows():
        # in some cases model generates a tie
        tie = (
            sum(
                [
                    df.columns[ind].startswith("result:") and value == 1
                    for ind, value in enumerate(row)
                ]
            )
            == C
        )

        for ind, value in enumerate(row):
            if df.columns[ind].startswith("result:"):
                assert value in [0, 1]
                if tie:
                    count[df.columns[ind]] += value / C
                else:
                    count[df.columns[ind]] += value

    for k in count:
        print(f"{k}: {count[k] / N * 100:.2f}%")


def visualize_opencole_rating(df: pd.DataFrame, id_key: str) -> None:
    df = df.drop(columns=[id_key])
    for name, series in df.items():
        mean = series.mean().item()
        std = series.std().item()
        print(f"{name}: {mean:.2f} ± {std:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, required=True)
    parser.add_argument("--csv_path", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path, dtype={"id": object})
    if args.type == "rating":
        visualize_rating_results(df, id_key="id")
    elif args.type == "voting":
        visualize_voting_results(df)
    elif args.type == "opencolerating":
        visualize_opencole_rating(df, id_key="id")
    else:
        raise NotImplementedError
