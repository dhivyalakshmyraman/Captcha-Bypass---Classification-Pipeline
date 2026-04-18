# Captcha-Bypass-Classification-Pipeline
Captcha Bypass &amp; Bot Classification with ML

# ML-Based Risk Detection & Decision Engine

**Project Status:** In Progress  
**Current Stage:** Literature Survey

---

## Overview

This project implements an end-to-end **ML-driven risk detection and decision system** using passive client-side signals.  
The system ingests behavioral and contextual features, applies machine learning–based detection, computes risk scores, and enforces decisions at a protected API boundary.

---

## Technology Stack

- **Backend / API:** FastAPI  
- **ML Platform:** Vertex AI  
- **Experiment Tracking & Reproducibility:** MLflow + DVC  
- **Data & Model Validation:** Deepchecks  

---

## System Architecture


Client (Browser / App)
↓ (Passive signals)
Feature Collector (JS / SDK)
↓
Feature Engineering Layer
↓
ML Detection Engine
↓
Risk Scoring Engine
↓
Decision Engine
↓
Protected API (Allow / Challenge / Block)


---

## Component Breakdown

### 1. Client (Browser / App)
- Generates passive behavioral and contextual signals (e.g., interaction timing, device hints).
- Designed to avoid explicit user friction.

### 2. Feature Collector (JS / SDK)
- Securely captures raw passive signals.
- Performs lightweight preprocessing and batching.
- Sends data to backend ingestion endpoints.

### 3. Feature Engineering Layer
- Cleans, normalizes, and aggregates raw signals.
- Transforms events into model-ready features.
- Fully versioned using **DVC** for reproducibility.

### 4. ML Detection Engine
- Applies trained ML models to detect anomalies, abuse, or suspicious behavior.
- Models are trained and deployed using **Vertex AI**.
- Experiments, metrics, and artifacts tracked via **MLflow**.

### 5. Risk Scoring Engine
- Converts model outputs into calibrated risk scores.
- Supports thresholding, weighting, and ensemble logic.

### 6. Decision Engine
- Applies deterministic business rules on top of risk scores.
- Produces one of the following outcomes:
  - **Allow**
  - **Challenge**
  - **Block**

### 7. Protected API
- Enforces final decisions at the API boundary.
- Integrates with authentication, authorization, and rate-limiting mechanisms.

---

## Quality & Validation

**Deepchecks** is used to ensure:
- Data integrity and consistency
- Model performance validation
- Drift detection
- Bias and edge-case analysis

---

## Current Focus

- Conducting a comprehensive literature survey on:
  - Passive signal–based detection
  - Risk scoring methodologies
  - ML-based abuse and anomaly detection

---

## Upcoming Milestones

- Complete literature survey
- Finalize feature taxonomy
- Implement baseline detection models
- Define and validate end-to-end evaluation metrics

---

## Notes

This repository is under active development.  
Design choices, models, and interfaces may evolve as research progresses.

