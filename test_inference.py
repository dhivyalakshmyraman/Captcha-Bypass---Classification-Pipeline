import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api import assess
from src.api.schemas import SessionAssessmentRequest

async def run_test():
    print("--- Initializing Models ---")
    processed_dir = PROJECT_ROOT / "data" / "processed"
    assess.load_models(processed_dir)

    # 1. SAMPLE HUMAN INPUT
    human_request = SessionAssessmentRequest(
        session_id="test_human_001",
        keystroke={
            "hold_times": [0.12, 0.11, 0.13, 0.10, 0.12],
            "flight_times": [0.25, 0.28, 0.22, 0.26, 0.24],
            "digraph_times": [0.37, 0.39, 0.35, 0.36, 0.36]
        },
        mouse={
            "timestamps": [0, 100, 200, 300, 400],
            "x": [100, 105, 115, 130, 150],
            "y": [100, 102, 108, 118, 132]
        },
        fingerprint={
            "canvas_hash": "a1b2c3d4",
            "webgl_hash": "e5f6g7h8",
            "webdriver": False
        },
        network={
            "flow_duration": 500000,
            "fwd_pkt_count": 10,
            "bwd_pkt_count": 8,
            "iat_mean": 5000
        },
        webbot={
            "request_rate": 0.5,
            "js_enabled": 1,
            "referrer_present": 1.0
        }
    )

    # 2. SAMPLE BOT INPUT (Fast periodic timing, automated signatures)
    bot_request = SessionAssessmentRequest(
        session_id="test_bot_001",
        keystroke={
            "hold_times": [0.01, 0.01, 0.01, 0.01, 0.01],
            "flight_times": [0.01, 0.01, 0.01, 0.01, 0.01],
            "digraph_times": [0.02, 0.02, 0.02, 0.02, 0.02]
        },
        mouse={
            "timestamps": [0, 10, 20, 30, 40],
            "x": [100, 200, 300, 400, 500],
            "y": [100, 200, 300, 400, 500]
        },
        fingerprint={
            "canvas_hash": "bot_hash_999",
            "webdriver": True
        },
        network={
            "flow_duration": 1000,
            "fwd_pkt_count": 500,
            "bwd_pkt_count": 500,
            "iat_mean": 1
        },
        webbot={
            "request_rate": 100.0,
            "js_enabled": 0,
            "error_rate": 0.8
        }
    )

    print("\n--- Testing HUMAN Input ---")
    risk_h, decision_h = await assess.assess_session_async(human_request)
    print(f"Outcome: Risk={risk_h}, Decision={decision_h}")

    print("\n--- Testing BOT Input ---")
    risk_b, decision_b = await assess.assess_session_async(bot_request)
    print(f"Outcome: Risk={risk_b}, Decision={decision_b}")

if __name__ == "__main__":
    asyncio.run(run_test())
