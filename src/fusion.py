from pathlib import Path
import json, joblib
import numpy as np, pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from src.llm_features import feature_rows

D=Path("data/processed/hybrid"); M=Path("models/fusion"); R=Path("results")
def get_classical_probs(dataset, train, valid, test):
    # Train each base learner on training data; obtain out-of-sample probabilities
    # for validation and test. Validation probabilities train the fusion layer.
    vec=TfidfVectorizer(ngram_range=(1,2),max_features=100000,min_df=2,sublinear_tf=True,stop_words="english")
    a=vec.fit_transform(train.text.astype(str)); b=vec.transform(valid.text.astype(str)); c=vec.transform(test.text.astype(str))
    models=[
      LogisticRegression(max_iter=1200,class_weight="balanced",C=2),
      CalibratedClassifierCV(LinearSVC(class_weight="balanced"),cv=3),
      RandomForestClassifier(n_estimators=180,min_samples_leaf=2,class_weight="balanced_subsample",n_jobs=-1,random_state=42)
    ]
    pv=[]; pt=[]
    for model in models:
        model.fit(a,train.label.astype(int))
        pv.append(model.predict_proba(b)[:,1]); pt.append(model.predict_proba(c)[:,1])
    return np.column_stack(pv),np.column_stack(pt),vec
def run(dataset="isot", llm_limit=None):
    M.mkdir(parents=True,exist_ok=True); R.mkdir(exist_ok=True)
    tr=pd.read_csv(D/f"{dataset}_train.csv").fillna("")
    va=pd.read_csv(D/f"{dataset}_valid.csv").fillna("")
    te=pd.read_csv(D/f"{dataset}_test.csv").fillna("")
    # Limit expensive LLM calls deterministically; unprocessed rows are explicitly
    # represented as missing/neutral and flagged. Do not treat limited run as full.
    vtexts=va.text.astype(str).tolist(); ttexts=te.text.astype(str).tolist()
    vlim=len(vtexts) if llm_limit is None else min(llm_limit,len(vtexts))
    tlim=len(ttexts) if llm_limit is None else min(llm_limit,len(ttexts))
    lv=np.zeros((len(va),6),dtype=float); lt=np.zeros((len(te),6),dtype=float)
    lv[:,0]=.5; lt[:,0]=.5; lv[:,4]=1; lt[:,4]=1
    if vlim:
        lv[:vlim]=np.asarray(feature_rows(vtexts,limit=vlim))
    if tlim:
        lt[:tlim]=np.asarray(feature_rows(ttexts,limit=tlim))
    pv,pt,vec=get_classical_probs(dataset,tr,va,te)
    # Train fusion on validation features, evaluate on held-out test.
    Xv=np.column_stack([pv,lv]); Xt=np.column_stack([pt,lt])
    yv=va.label.astype(int).to_numpy(); yt=te.label.astype(int).to_numpy()
    fusion=XGBClassifier(n_estimators=250,max_depth=3,learning_rate=.04,
        subsample=.85,colsample_bytree=.9,reg_lambda=2,eval_metric="logloss",
        random_state=42,n_jobs=2)
    fusion.fit(Xv,yv)
    prob=fusion.predict_proba(Xt)[:,1]; pred=(prob>=.5).astype(int)
    pr,re,f1,_=precision_recall_fscore_support(yt,pred,average="binary",zero_division=0)
    metrics={"accuracy":float(accuracy_score(yt,pred)),"precision":float(pr),
             "recall":float(re),"f1":float(f1),"roc_auc":float(roc_auc_score(yt,prob)),
             "test_rows":len(te),"llm_valid_rows":int(lv[:vlim,5].sum()),"llm_test_rows":int(lt[:tlim,5].sum()),"llm_valid_attempted":vlim,"llm_test_attempted":tlim,
             "llm_model":__import__("os").getenv("GROQ_MODEL","openai/gpt-oss-20b")}
    joblib.dump({"fusion":fusion,"tfidf":vec},M/f"{dataset}_hybrid.joblib")
    (R/f"{dataset}_hybrid.json").write_text(json.dumps(metrics,indent=2))
    np.savez(R/f"{dataset}_fusion_features.npz",X_test=Xt,y_test=yt)
    print(json.dumps(metrics,indent=2))
    return metrics
