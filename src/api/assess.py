import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import pandas as pd
import numpy as np

# Load local modules
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Assuming we have the transform_online functions in features
from src.features import ks_pipeline
from src.features import ms_pipeline
from src.features import fp_pipeline
from src.features import net_pipeline
from src.features import wb_pipeline
from src.api.schemas import SessionAssessmentRequest

logger = logging.getLogger(__name__)

# Global cache for models
MODELS = {}

def load_models(processed_dir: Path):
    """Load all 6 models into memory"""
    global MODELS
    
    signals = ["ks", "ms", "fp", "net", "wb", "meta"]
    for sig in signals:
        path = processed_dir / f"{sig}_pipeline.joblib"
        if sig == "meta":
            path = processed_dir / "meta_model.joblib"
            
        if path.exists():
            MODELS[sig] = joblib.load(path)
            logger.info(f"Loaded {sig} model")
        else:
            logger.warning(f"Model missing: {path}")

def get_submodel_score(signal: str, raw_data: Dict[str, Any]) -> float:
    """Extract features, run prediction, return p(bot) [0.0 - 1.0]"""
    if raw_data is None:
        return 0.5
        
    if signal not in MODELS:
        logger.error(f"Model {signal} not loaded!")
        return 0.5
        
    try:
        if signal == "ks":
            pipeline = MODELS["ks"]
            scaler = pipeline.named_steps["scaler"]
            feats = ks_pipeline.transform_online(raw_data, scaler)
            proba = pipeline.named_steps["model"].predict_proba(feats)[0][1]
        elif signal == "ms":
            pipeline = MODELS["ms"]
            scaler = pipeline.named_steps["scaler"]
            feats = ms_pipeline.transform_online(raw_data, scaler)
            proba = pipeline.named_steps["model"].predict_proba(feats)[0][1]
        elif signal == "fp":
            pipeline = MODELS["fp"]
            scaler = pipeline.named_steps["scaler"]
            feats = fp_pipeline.transform_online(raw_data, scaler)
            proba = pipeline.named_steps["model"].predict_proba(feats)[0][1]
        elif signal == "net":
            data = MODELS["net"]
            scaler = data["scaler"]
            wrapper = data["wrapper"]
            feats = net_pipeline.transform_online(raw_data, scaler)
            proba = wrapper.predict_proba(feats)[0][1]
        elif signal == "wb":
            pipeline = MODELS["wb"]
            scaler = pipeline.named_steps["scaler"]
            feats = wb_pipeline.transform_online(raw_data, scaler)
            proba = pipeline.named_steps["model"].predict_proba(feats)[0][1]
        else:
            return 0.5
            
        return float(proba)
    except Exception as e:
        logger.error(f"Error scoring {signal}: {e}")
        return 0.5

async def assess_session_async(request: SessionAssessmentRequest) -> Tuple[int, str]:
    """Run all models concurrently and compute meta-score"""
    # 1. Run Sub Models concurrently
    tasks = [
        asyncio.to_thread(get_submodel_score, "ks", request.keystroke),
        asyncio.to_thread(get_submodel_score, "ms", request.mouse),
        asyncio.to_thread(get_submodel_score, "fp", request.fingerprint),
        asyncio.to_thread(get_submodel_score, "net", request.network),
        asyncio.to_thread(get_submodel_score, "wb", request.webbot),
    ]
    
    scores = await asyncio.gather(*tasks)
    p_ks, p_ms, p_fp, p_net, p_wb = scores
    
    logger.debug(f"{request.session_id} raw sub-scores: ks={p_ks}, ms={p_ms}, fp={p_fp}, net={p_net}, wb={p_wb}")
    
    # 2. Run Meta Model
    meta_model = MODELS.get("meta")
    if meta_model is None:
        logger.error("Meta model not found")
        # Simple fallback
        risk_float = float(np.mean([s for s in scores if s != 0.5] or [0.5]))
    else:
        # Note: sub-models return p(human) usually, but net_model returns p(bot)? 
        # Actually our submodels return p(human=1).
        meta_feats = np.array([[p_ks, p_ms, p_fp, p_net, p_wb]])
        # predict_proba returns [p(bot=0), p(human=1)] generally or vice versa depending on meta-learner training
        # Assuming label 1 = human, label 0 = bot
        # So we want risk to be p(bot) -> index 0 from predict_proba if 0 is first class
        
        # We need risk_float as a bot probability
        meta_proba = meta_model.predict_proba(meta_feats)[0]
        # usually class 0 is bot, class 1 is human
        classes = meta_model.classes_
        bot_idx = list(classes).index(0) if 0 in classes else 0
        risk_float = meta_proba[bot_idx]
        
    risk_score = int(risk_float * 100)
    
    # 3. Decision Logic
    if risk_score <= 39:
        decision = "allow"
    elif risk_score <= 69:
        decision = "stepup"
    else:
        decision = "blocked"
        
    # Log to CSV (placeholder for BigQuery)
    # Using local CSV for simplicity
    log_line = f"{request.session_id},{risk_score},{decision},{int(request.keystroke is not None)},{int(request.mouse is not None)},{int(request.fingerprint is not None)},{int(request.network is not None)},{int(request.webbot is not None)}\n"
    with open("assessment_log.csv", "a") as f:
        f.write(log_line)
        
    return risk_score, decision
