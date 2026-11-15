from pathlib import Path
import pandas as pd
for p in sorted(Path("data/raw/isot").glob("*.csv")):
    d=pd.read_csv(p); print(p,len(d),list(d.columns))
for p in sorted(Path("data/raw/liar").glob("*.tsv")):
    print(p, sum(1 for _ in open(p,encoding="utf-8")))
