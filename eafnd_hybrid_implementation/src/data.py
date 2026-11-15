from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

RAW = Path("data/raw")
OUT = Path("data/processed/hybrid")
SEED = 42
LABEL_MAP = {
    "true": 0, "mostly-true": 0,
    "half-true": 1, "barely-true": 1, "false": 1, "pants-fire": 1
}
LIAR_COLS = [
    "id","label_text","statement","subject","speaker","speaker_job",
    "state","party","barely_true_counts","false_counts","half_true_counts",
    "mostly_true_counts","pants_fire_counts","venue"
]

def load_isot(sample_per_class=5000):
    real = pd.read_csv(RAW/"isot/True.csv").fillna("")
    fake = pd.read_csv(RAW/"isot/Fake.csv").fillna("")
    real["label"] = 0
    fake["label"] = 1
    real = real.sample(n=min(sample_per_class,len(real)), random_state=SEED)
    fake = fake.sample(n=min(sample_per_class,len(fake)), random_state=SEED)
    df = pd.concat([real,fake], ignore_index=True)
    df["text"] = (df["title"].astype(str) + ". " + df["text"].astype(str)).str.strip()
    df["dataset"] = "ISOT"
    return df[["text","label","dataset"]].drop_duplicates("text").reset_index(drop=True)

def load_liar_split(name):
    path = RAW/"liar"/f"{name}.tsv"
    df = pd.read_csv(path, sep="\t", header=None, names=LIAR_COLS, quoting=3,
                     on_bad_lines="skip", dtype=str).fillna("")
    df["label"] = df["label_text"].str.lower().map(LABEL_MAP)
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    df["text"] = df["statement"].astype(str).str.strip()
    df["dataset"] = "LIAR"
    return df[["text","label","dataset"]].drop_duplicates("text").reset_index(drop=True)

def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    isot = load_isot()
    train, temp = train_test_split(isot, test_size=.30, random_state=SEED, stratify=isot.label)
    valid, test = train_test_split(temp, test_size=.50, random_state=SEED, stratify=temp.label)
    for name, frame in [("isot_train",train),("isot_valid",valid),("isot_test",test)]:
        frame.to_csv(OUT/f"{name}.csv", index=False)
    liar = {s:load_liar_split(s) for s in ("train","valid","test")}
    for s, frame in liar.items():
        frame.to_csv(OUT/f"liar_{s}.csv", index=False)
    return {
        "isot": {k:len(v) for k,v in [("train",train),("valid",valid),("test",test)]},
        "liar": {k:len(v) for k,v in liar.items()}
    }

if __name__ == "__main__":
    import json
    print(json.dumps(prepare(), indent=2))
