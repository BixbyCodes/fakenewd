from pathlib import Path
import json, joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

DATA=Path("data/processed/hybrid"); MODELS=Path("models/classical"); RESULTS=Path("results")
def metrics(y,p,proba=None):
    pr,re,f1,_=precision_recall_fscore_support(y,p,average="binary",zero_division=0)
    out={"accuracy":float(accuracy_score(y,p)),"precision":float(pr),"recall":float(re),"f1":float(f1)}
    if proba is not None:
        try: out["roc_auc"]=float(roc_auc_score(y,proba))
        except ValueError: out["roc_auc"]=None
    return out
def run(dataset="isot"):
    MODELS.mkdir(parents=True,exist_ok=True); RESULTS.mkdir(exist_ok=True)
    tr=pd.read_csv(DATA/f"{dataset}_train.csv").fillna("")
    va=pd.read_csv(DATA/f"{dataset}_valid.csv").fillna("")
    te=pd.read_csv(DATA/f"{dataset}_test.csv").fillna("")
    vec=TfidfVectorizer(ngram_range=(1,2),max_features=150000,min_df=2,
                        sublinear_tf=True,strip_accents="unicode",stop_words="english")
    Xtr=vec.fit_transform(tr.text.astype(str)); Xva=vec.transform(va.text.astype(str)); Xte=vec.transform(te.text.astype(str))
    models={
      "logistic_regression":LogisticRegression(max_iter=1500,class_weight="balanced",C=2.0),
      "linear_svm":CalibratedClassifierCV(LinearSVC(class_weight="balanced",C=1.0),cv=3),
      "random_forest":RandomForestClassifier(n_estimators=250,max_depth=None,min_samples_leaf=2,
          class_weight="balanced_subsample",n_jobs=-1,random_state=42)
    }
    report={}
    for name,model in models.items():
        model.fit(Xtr,tr.label.astype(int))
        pred=model.predict(Xte)
        prob=model.predict_proba(Xte)[:,1] if hasattr(model,"predict_proba") else None
        report[name]=metrics(te.label.astype(int),pred,prob)
        joblib.dump(model,MODELS/f"{dataset}_{name}.joblib")
        np.save(MODELS/f"{dataset}_{name}_test_proba.npy",
                prob if prob is not None else pred.astype(float))
        if name=="logistic_regression":
            np.save(MODELS/f"{dataset}_test_tfidf_features.npy",Xte.toarray().astype("float32"))
        print(name,report[name])
    joblib.dump(vec,MODELS/f"{dataset}_tfidf.joblib")
    (RESULTS/f"{dataset}_classical.json").write_text(json.dumps(report,indent=2))
    return report
if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(); p.add_argument("--dataset",choices=["isot","liar"],default="isot")
    run(p.parse_args().dataset)
