import time
from datetime import datetime, timedelta
from jose import jwt
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import logging
from pathlib import Path

# Local modules
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.schemas import SessionAssessmentRequest, SessionAssessmentResponse
from src.api.assess import load_models, assess_session_async

# Conf
SECRET_KEY = "my_super_secret_for_jwt_signing"
ALGORITHM = "HS256" # Should be RS256 with keys in prod

# Simple app
app = FastAPI(title="Bot Detection API")

@app.on_event("startup")
async def startup_event():
    logging.basicConfig(level=logging.INFO)
    processed_dir = PROJECT_ROOT / "data" / "processed"
    logging.info(f"Loading models from {processed_dir}")
    load_models(processed_dir)
    # Init blank log file
    if not Path("assessment_log.csv").exists():
        with open("assessment_log.csv", "w") as f:
            f.write("session_id,risk_score,decision,has_ks,has_ms,has_fp,has_net,has_wb\n")

@app.post("/v1/assess-session", response_model=SessionAssessmentResponse)
async def assess_session(request: SessionAssessmentRequest):
    start = time.time()
    
    # Run assessment
    risk_score, decision = await assess_session_async(request)
    
    response = {"decision": decision, "risk_score": risk_score}
    
    if decision == "allow":
        # Generate token
        expires = datetime.utcnow() + timedelta(minutes=15)
        payload = {"sub": request.session_id, "exp": expires}
        token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
        response["token"] = token
    elif decision == "stepup":
        response["challenge"] = "otp"
    else:
        raise HTTPException(status_code=403, detail="blocked")
        
    elapsed = time.time() - start
    logging.info(f"Session {request.session_id} assessed in {elapsed*1000:.2f}ms. Risk: {risk_score}, Dec: {decision}")
    
    return JSONResponse(status_code=200, content=response)
