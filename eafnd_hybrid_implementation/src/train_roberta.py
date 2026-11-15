from pathlib import Path
import argparse, numpy as np, pandas as pd
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

D=Path("data/processed/hybrid"); OUT=Path("models/roberta")
def main(epochs=2, dataset="isot"):
    OUT.mkdir(parents=True,exist_ok=True)
    tr=pd.read_csv(D/f"{dataset}_train.csv").fillna("")
    va=pd.read_csv(D/f"{dataset}_valid.csv").fillna("")
    tokenizer=AutoTokenizer.from_pretrained("roberta-base")
    def encode(batch): return tokenizer(batch["text"],truncation=True,max_length=256)
    ds={}
    for name,df in [("train",tr),("valid",va)]:
        d=Dataset.from_pandas(df[["text","label"]])
        ds[name]=d.map(encode,batched=True).rename_column("label","labels")
        ds[name].set_format(type="torch",columns=["input_ids","attention_mask","labels"])
    model=AutoModelForSequenceClassification.from_pretrained("roberta-base",num_labels=2)
    args=TrainingArguments(output_dir=str(OUT/"checkpoints"),num_train_epochs=epochs,
      per_device_train_batch_size=2,per_device_eval_batch_size=2,
      gradient_accumulation_steps=8,learning_rate=2e-5,weight_decay=.01,
      evaluation_strategy="epoch",save_strategy="epoch",load_best_model_at_end=True,
      metric_for_best_model="f1",fp16=False,report_to="none",logging_steps=100)
    def compute(p):
        logits,labels=p; pred=np.argmax(logits,axis=1)
        pr,re,f1,_=precision_recall_fscore_support(labels,pred,average="binary",zero_division=0)
        return {"accuracy":accuracy_score(labels,pred),"f1":f1,"precision":pr,"recall":re}
    trainer=Trainer(model=model,args=args,train_dataset=ds["train"],eval_dataset=ds["valid"],
      tokenizer=tokenizer,data_collator=DataCollatorWithPadding(tokenizer),compute_metrics=compute)
    trainer.train(); trainer.save_model(str(OUT/dataset)); tokenizer.save_pretrained(str(OUT/dataset))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--epochs",type=int,default=2);p.add_argument("--dataset",default="isot")
    a=p.parse_args();main(a.epochs,a.dataset)
