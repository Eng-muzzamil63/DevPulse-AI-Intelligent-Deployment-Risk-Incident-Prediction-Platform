from pathlib import Path
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "xgboost_model.joblib"
DEPLOYMENTS = ROOT / "data" / "deployments.json"
FEATURES = ["files_changed","lines_added","lines_deleted","test_coverage","developer_experience","previous_incidents_30d","rollback_rate_90d","deployment_frequency_7d","database_changes","services_touched","hour_of_day"]

app = FastAPI(title="DevPulse AI API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173","http://127.0.0.1:5173"], allow_methods=["*"], allow_headers=["*"])

class Payload(BaseModel):
    features: dict

def model():
    if not MODEL.exists(): raise RuntimeError("Model not found. Run python ml/train.py")
    return joblib.load(MODEL)

def prep(f):
    return pd.DataFrame([{k: f.get(k, 0) for k in FEATURES}], columns=FEATURES)

def label(p):
    return "Critical" if p >= .80 else "High" if p >= .60 else "Medium" if p >= .30 else "Low"

def explain(f):
    out=[]
    checks=[
        ("test_coverage", float(f.get("test_coverage",0))<75, "high", f"Test coverage is {float(f.get('test_coverage',0)):.0f}%."),
        ("files_changed", int(f.get("files_changed",0))>=35, "high", f"{int(f.get('files_changed',0))} files are changing."),
        ("previous_incidents_30d", int(f.get("previous_incidents_30d",0))>=2, "high", f"{int(f.get('previous_incidents_30d',0))} incidents occurred in the last 30 days."),
        ("rollback_rate_90d", float(f.get("rollback_rate_90d",0))>=.20, "high", f"Historical rollback rate is {float(f.get('rollback_rate_90d',0))*100:.0f}%."),
        ("database_changes", int(f.get("database_changes",0))>0, "medium", f"{int(f.get('database_changes',0))} database change(s) are included."),
    ]
    for name, yes, impact, reason in checks:
        if yes: out.append({"feature":name,"impact":impact,"reason":reason})
    return out[:5] or [{"feature":"baseline","impact":"low","reason":"No major risk threshold was crossed."}]

def recs(f,p):
    r=[]
    r.append({"title":"Stage before production" if p>=.8 else "Run targeted validation" if p>=.6 else "Monitor deployment","detail":"Use a production-like staging environment and the full regression suite." if p>=.8 else "Run integration tests for affected services before release." if p>=.6 else "Proceed with standard monitoring and rollback readiness.","priority":"Critical" if p>=.8 else "High" if p>=.6 else "Normal"})
    if float(f.get("test_coverage",0))<80: r.append({"title":"Increase test coverage","detail":"Add automated tests around changed modules.","priority":"High"})
    if int(f.get("database_changes",0))>0: r.append({"title":"Validate database migration","detail":"Test migration and rollback safety on a production-like snapshot.","priority":"High"})
    if int(f.get("previous_incidents_30d",0))>=2: r.append({"title":"Require senior review","detail":"Recent instability makes an additional engineering review worthwhile.","priority":"High"})
    return r[:4]

@app.get("/api/health")
def health(): return {"status":"ok"}

@app.get("/api/dashboard")
def dashboard():
    rows=json.loads(DEPLOYMENTS.read_text())
    avg=sum(x["risk_probability"] for x in rows)/len(rows)
    high=sum(x["risk_probability"]>=.60 for x in rows)
    critical=sum(x["risk_probability"]>=.80 for x in rows)
    incidents=sum(x["status"]=="incident" for x in rows)
    services={}
    for x in rows: services.setdefault(x["service"],[]).append(x)
    health=[{"service":s,"health":round((1-sum(x["risk_probability"] for x in xs)/len(xs))*100),"deployments":len(xs)} for s,xs in services.items()]
    return {"deployments":len(rows),"high_risk":high,"critical_risk":critical,"incidents":incidents,"avg_risk":round(avg*100,1),"service_health":health,"risk_trend":[{"date":x["started_at"],"risk":round(x["risk_probability"]*100,1)} for x in rows]}

@app.get("/api/deployments")
def deployments(): return json.loads(DEPLOYMENTS.read_text())

@app.post("/api/predict")
def predict(payload: Payload):
    try:
        p=float(model().predict_proba(prep(payload.features))[0][1])
        return {"risk_probability":round(p,4),"risk_label":label(p),"top_risk_factors":explain(payload.features),"recommendations":recs(payload.features,p)}
    except Exception as e: raise HTTPException(500,str(e))
