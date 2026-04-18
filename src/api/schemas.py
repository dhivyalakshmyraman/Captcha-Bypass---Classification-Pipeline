from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class KeystrokeData(BaseModel):
    hold_times: Optional[List[float]] = None
    flight_times: Optional[List[float]] = None
    digraph_times: Optional[List[float]] = None
    # Add explicit H.* and UD.* if needed, but dict form handles it

class MouseData(BaseModel):
    timestamps: Optional[List[float]] = None
    x: Optional[List[int]] = None
    y: Optional[List[int]] = None

class FingerprintData(BaseModel):
    user_agent: Optional[str] = None
    canvas_hash: Optional[str] = None
    webgl_hash: Optional[str] = None
    timezone: Optional[str] = None
    language_count: Optional[int] = 1
    plugin_count: Optional[int] = 4
    screen_width: Optional[int] = 1920
    touch_support: Optional[int] = 0
    webdriver: Optional[bool] = False
    font_count: Optional[int] = 8

class NetworkData(BaseModel):
    flow_duration: Optional[float] = 0.0
    fwd_pkt_count: Optional[int] = 0
    bwd_pkt_count: Optional[int] = 0
    flow_bytes_per_sec: Optional[float] = 0.0
    flow_pkts_per_sec: Optional[float] = 0.0
    iat_mean: Optional[float] = 0.0
    iat_std: Optional[float] = 0.0

class WebBotData(BaseModel):
    user_agent: Optional[str] = ""
    request_rate: Optional[float] = 1.0
    method_diversity: Optional[int] = 1
    js_enabled: Optional[int] = 1
    timing_regularity: Optional[float] = 1.0
    referrer_present: Optional[float] = 0.5
    unique_urls: Optional[int] = 5
    avg_response_size: Optional[float] = 1000.0
    error_rate: Optional[float] = 0.0

class SessionAssessmentRequest(BaseModel):
    session_id: str
    keystroke: Optional[Dict[str, Any]] = None
    mouse: Optional[Dict[str, Any]] = None
    fingerprint: Optional[Dict[str, Any]] = None
    network: Optional[Dict[str, Any]] = None
    webbot: Optional[Dict[str, Any]] = None

class SessionAssessmentResponse(BaseModel):
    decision: str
    risk_score: int
    token: Optional[str] = None
    challenge: Optional[str] = None
