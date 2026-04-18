# Multi-Signal Bot Detection

This project implements a multi-modal bot detection system using Keystroke Dynamics, Mouse Dynamics, Browser Fingerprinting, Network Traffic, and Web Session Logs. 

## Structure

- `src/features/` - Feature extraction pipelines (ks, ms, fp, net, wb)
- `src/models/` - Sub-model training (RandomForest, XGBoost, IsolationForest) and Meta-Learner (LogisticRegression)
- `src/gan/` - Adversarial training (Generator, Discriminator) to harden models via GANs
- `src/api/` - FastAPI service for real-time assessment
- `src/validation/` - Deepchecks data integrity validation

## Quickstart

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run full MLOps pipeline via DVC:
```bash
dvc repro
```

3. Run API Server:
```bash
uvicorn src.api.main:app --reload
```

## Sample Inference

```bash
curl -X POST "http://localhost:8000/v1/assess-session" \
     -H "Content-Type: application/json" \
     -d '{
           "session_id": "test_session_1",
           "fingerprint": {
               "user_agent": "Mozilla/5.0",
               "canvas_hash": "a1b2c3d4"
           }
         }'
```
