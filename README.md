# Milliqan Anomaly Detection Pipeline

Author: Saloni Shah

## Overview

This repository implements an end-to-end anomaly detection pipeline for the **Milliqan experiment**. The system analyzes pulse-level detector data and identifies detector channels that exhibit abnormal signal behavior.

The pipeline processes new run data, applies preprocessing and feature transformations, evaluates pulses using a **conditional autoencoder**, and aggregates anomaly statistics at the channel level.

Channels with unusually high fractions of anomalous pulses are flagged for further inspection. Results are exported and visualized through a **Streamlit dashboard** for monitoring detector health.

---

## Pipeline

The workflow is:

Run Data → Preprocessing → Feature Engineering → Scaling
→ Conditional Autoencoder → Reconstruction Loss
→ Channel Metrics → Anomaly Rating → Output Files → Dashboard


Steps performed:

1. Load detector run data
2. Apply detector quality cuts and preprocessing
3. Engineer and transform features
4. Normalize features using a saved scaler
5. Run inference using a trained conditional autoencoder
6. Compute reconstruction loss for each pulse
7. Aggregate anomaly statistics by detector channel
8. Assign anomaly severity ratings
9. Export pulse-level and channel-level outputs
10. Visualize results in the Streamlit monitoring dashboard

---

## Model

The anomaly detector is a **conditional autoencoder** implemented in TensorFlow/Keras.  
The model reconstructs pulse features while conditioning on the detector **channel ID** using a learned embedding.

### Encoder

Input features + channel embedding  
→ Dense(64)  
→ Dense(32)  
→ Latent representation (16)

### Decoder

Latent + channel embedding  
→ Dense(32)  
→ Dense(64)  
→ Output reconstruction

Key details:

- Channel embedding dimension: **13**
- Loss: **Mean Squared Error**
- Optimizer: **Adam**
- Gaussian input noise for regularization
- L2 weight decay
- Early stopping during training

---

## Anomaly Scoring

For each pulse, anomaly score is computed as the reconstruction error:
```
score = mean((x - x_reconstructed)^2)
```


Channel-specific thresholds are learned from the training set using the **99.5th percentile** of reconstruction error.

For each channel the pipeline computes:

- mean reconstruction error
- standard deviation of reconstruction error
- number of anomalous pulses
- fraction of anomalous pulses

Channels are assigned severity ratings:

| Fraction of Anomalous Pulses | Rating |
|-------------------------------|--------|
| < 1% | Low |
| ≥ 1% | Medium |
| ≥ 3% | High |

---

## Automation

The anomaly detection pipeline can run automatically on new detector runs.

A scheduled script: run_daily.sh

The script:
1. Reads the last processed run from a state file
2. Queries the Milliqan run directory for the latest available run 
3. Skips execution if no new run is available 
4. Constructs the pulse data URL for the new run 
5. Executes the anomaly detection pipeline 
6. Saves outputs under a run-specific output directory 
7. Writes logs for monitoring and debugging 
8. Updates the saved pipeline state 

The pipeline is run **once per day** on cms18.

---

## Dashboard

Results are visualized using a **Streamlit dashboard** that displays:

- channel anomaly ratings
- reconstruction loss distributions
- anomalous pulse fractions
- run-level detector diagnostics

Run the dashboard locally: streamlit run dashboard/app.py

## Repository Structure

artifacts/    saved scalers and thresholds  
dashboard/    Streamlit monitoring app  
data/         training data  
logs/         pipeline execution logs  
models/       trained autoencoder models  
outputs/      run-level anomaly results  
scripts/      preprocessing, training, and inference scripts  
run_daily.sh  automated pipeline runner