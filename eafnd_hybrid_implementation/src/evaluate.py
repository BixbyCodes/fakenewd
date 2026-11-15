from pathlib import Path
import json, numpy as np, pandas as pd, joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from src.fusion import run as run_fusion
from src.data import prepare

def robustness(texts, labels, model, vectorizer):
    variants={"original":list(texts),
      "truncation":[str(x)[:max(1,int(len(str(x))*.5))] for x in texts],
      "typos":[str(x)[:max(0,len(str(x))//2)]+" "+str(x)[len(str(x))//2:] .replace("e","3",1) for x in texts]}
    out={}
    for name,items in variants.items():
        out[name]=float(f1_score(labels,model.predict(vectorizer.transform(items))))
    return out

def main(dataset="isot"):
    from src.classical import run
    run(dataset)
    report={}
    d=Path("data/processed/hybrid"); r=Path("results")
    tr=pd.read_csv(d/f"{dataset}_train.csv").fillna("")
    te=pd.read_csv(d/f"{dataset}_test.csv").fillna("")
    model=joblib.load(Path("models/classical")/f"{dataset}_logistic_regression.joblib")
    vec=joblib.load(Path("models/classical")/f"{dataset}_tfidf.joblib")
    report["robustness"]=robustness(te.text.astype(str).tolist(),te.label.astype(int),model,vec)
    report["classification_report"]=classification_report(te.label,model.predict(vec.transform(te.text.astype(str))),output_dict=True)
    (r/f"{dataset}_evaluation.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--dataset",choices=["isot","liar"],default="isot")
    main(p.parse_args().dataset)
