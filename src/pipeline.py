import argparse, json, os
from pathlib import Path
from src.data import prepare
from src.classical import run as classical_run
from src.fusion import run as fusion_run

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",choices=["isot","liar"],default="isot")
    p.add_argument("--llm-limit",type=int,default=100)
    p.add_argument("--skip-roberta",action="store_true")
    p.add_argument("--use-roberta",action="store_true")
    a=p.parse_args()
    counts=prepare()
    print("DATA PREPARATION",json.dumps(counts,indent=2))
    print("\\nCLASSICAL MODELS")
    classical_run(a.dataset)
    print("\\nHYBRID FUSION")
    fusion_run(a.dataset,llm_limit=a.llm_limit)
    metadata={"dataset":a.dataset,"seed":42,"llm_limit_per_split":a.llm_limit,
      "llm_model":os.getenv("GROQ_MODEL","openai/gpt-oss-20b"),
      "roberta_included":False,
      "note":"RoBERTa training is a separate step; this run fuses TF-IDF classical model probabilities with LLM features."}
    Path("results").mkdir(exist_ok=True)
    Path("results/metadata.json").write_text(json.dumps(metadata,indent=2))
    print("\\nPIPELINE FINISHED. See results/ and models/.")
if __name__=="__main__": main()
