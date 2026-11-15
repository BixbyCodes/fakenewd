import os, json, sqlite3, hashlib, time
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
MODEL=os.getenv("GROQ_MODEL","openai/gpt-oss-20b")
DB=Path("data/processed/llm_features.sqlite")
SCHEMA={
 "type":"object",
 "properties":{
  "verdict":{"type":"string","enum":["real","fake","uncertain"]},
  "plausibility":{"type":"number"},
  "verifiability":{"type":"number"},
  "source_credibility":{"type":"number"},
  "rationale":{"type":"string"}
 },
 "required":["verdict","plausibility","verifiability","source_credibility","rationale"],
 "additionalProperties":False
}
def connect():
    DB.parent.mkdir(parents=True,exist_ok=True)
    c=sqlite3.connect(DB)
    c.execute("""CREATE TABLE IF NOT EXISTS features(
      text_hash TEXT PRIMARY KEY, text TEXT, verdict TEXT, plausibility REAL,
      verifiability REAL, source_credibility REAL, rationale TEXT, model TEXT,
      created_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
    return c
def extract_one(text, client):
    response=client.chat.completions.create(
      model=MODEL,temperature=0,max_completion_tokens=350,
      messages=[
       {"role":"system","content":
        "You are a cautious research annotation assistant. Assess only the supplied statement. "
        "Do not assume it is true or false without evidence. Use uncertain when the statement "
        "cannot be assessed from its wording alone. Scores are 0 to 1. Source credibility means "
        "whether a source is actually identified and assessable in the supplied text, not whether "
        "you personally trust an unnamed publisher. Return only schema-compliant values."},
       {"role":"user","content":text[:5000]}],
      response_format={"type":"json_schema","json_schema":{
        "name":"fake_news_features","strict":True,"schema":SCHEMA}})
    return json.loads(response.choices[0].message.content)
def feature_rows(texts, limit=None, delay=0.15):
    key=os.getenv("GROQ_API_KEY")
    if not key: raise RuntimeError("GROQ_API_KEY missing. Add it to .env.")
    client=Groq(api_key=key); c=connect(); out=[]
    subset=list(texts if limit is None else texts[:limit])
    for i,text in enumerate(subset,1):
        text=str(text); h=hashlib.sha256(text.encode()).hexdigest()
        row=c.execute("SELECT verdict,plausibility,verifiability,source_credibility FROM features WHERE text_hash=?",(h,)).fetchone()
        if row is None:
            try:
                d=extract_one(text,client)
                row=(d["verdict"],float(d["plausibility"]),float(d["verifiability"]),float(d["source_credibility"]))
                c.execute("INSERT OR REPLACE INTO features(text_hash,text,verdict,plausibility,verifiability,source_credibility,rationale,model) VALUES(?,?,?,?,?,?,?,?)",
                    (h,text,*row,d["rationale"],MODEL)); c.commit()
                time.sleep(delay)
            except Exception as e:
                print(f"LLM failed on row {i}: {e}")
                row=("uncertain",0.5,0.0,0.0)
        out.append([{"real":0.0,"fake":1.0,"uncertain":0.5}[row[0]],row[1],row[2],row[3],1.0 if row[0]=="uncertain" else 0.0])
        if i%25==0: print(f"LLM features: {i}/{len(subset)}")
    c.close()
    return out
