# 🚨 Women’s Safety ML System: Predictive Crime Analytics
> **Production-grade machine learning system for district-level crime forecasting across India.**
> ⚠️ *Note: This is a full-stack ML Engineering project featuring automated pipelines and live API deployment—not a standalone Jupyter Notebook.*

[![Live App](https://img.shields.io/badge/Demo-Live%20App-informational)](https://women-safety-ml-system-jwlh.vercel.app/)
[![API Docs](https://img.shields.io/badge/API-FastAPI%20Docs-success)](https://women-safety-ml-system.onrender.com/docs)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)

## 📌 Executive Summary
This project implements an end-to-end ML system to forecast district-level crime trends against women. Using 14 years of longitudinal NCRB data (2001–2014), the system utilizes **XGBoost-based time-series regression** to predict percentage changes in crime for the 2014–2015 period. 

By shifting from absolute counts to **Relative Risk Levels (Quartile-based)**, the system identifies emerging hotspots where crime is rising faster than the national median, providing actionable insights for policy intervention.

---

## ⚡ Key Engineering Highlights
* **Leakage-Proof Pipeline:** Implements a sliding-window time-series approach to prevent "look-ahead" bias.
* **Production Inference:** Model served via a **FastAPI** backend with asynchronous request handling.
* **Geospatial Intelligence:** Automated fuzzy-matching engine to reconcile disparate district naming conventions between NCRB data and GeoJSON boundaries.
* **High Performance:** Achieved an **$R^2 \approx 0.89$** on future unseen test data.

---

## 🏗️ System Architecture
The system is designed with modularity in mind, separating the concerns of data engineering, model inference, and visualization.



1.  **Data Layer:** Preprocessing of NCRB CSV data, outlier clipping, and feature scaling.
2.  **Model Layer:** XGBoost regressor trained on multi-year crime momentum and volatility features.
3.  **API Layer:** RESTful endpoints for real-time risk classification.
4.  **UI Layer:** React dashboard with Folium-powered interactive choropleth maps.

---

## 📊 Risk Classification Logic
To provide a standard for "Relative Risk," districts are ranked nationally using a **Quartile Distribution** of the predicted crime change.

| Range (Percentile) | Threat Level | Interpretation |
| :--- | :--- | :--- |
| **0 - 50% ($< Q_1$)** | <span style="color:green">**Low Risk**</span> | Districts with the highest improvement or lowest growth. |
| **50 - 75% ($Q_1$ to $Q_2$)** | <span style="color:orange">**Moderate Risk**</span> | Growth rate around the national median. |
| **75 - 90% ($Q_2$ to $Q_3$)** | <span style="color:red">**High Risk**</span> | Significant upward trend relative to the nation. |
| **Top 10% ($> Q_3$)** | <span style="color:darkred">**Very High Risk**</span> | Critical hotspots requiring immediate attention. |

> **Note on Negative Values:** If a district shows a $-5\%$ reduction in crime but the national median improvement is $-15\%$, this district will be classified as **Moderate Risk**. This ensures safety is measured against the best-performing regions.

---

## 🧠 The "Real-World" Difference
Most portfolio projects fail because they leak data or ignore deployment. This project addresses:
* **Time-Series Integrity:** We don't just shuffle rows. We predict 2015 using *only* data available prior to 2015.
* **Fuzzy Spatial Merging:** Developed a `rapidfuzz` pipeline to handle the 40% mismatch rate between government data and GIS maps.
* **Cold-Start Resilience:** Logic built to handle Render/Vercel free-tier spin-up delays for a seamless user experience.

---

## 🔬 Tech Stack
* **Machine Learning:** XGBoost, Scikit-learn, Pandas, Joblib.
* **Geospatial:** GeoPandas, Folium, Branca.
* **Backend:** FastAPI, Uvicorn.
* **Frontend:** React, Tailwind CSS, Lucide Icons.
* **Deployment:** Vercel (Frontend), Render (API).

---

## 🚀 Installation & Local Development
1. Clone the repo: `git clone`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the API: `uvicorn main:app --reload`
4. Run the UI: `npm start`

---

## 👨‍💻 Author
**Sanchit Rastogi** *B.Tech in Computer Science | ML Engineer* [LinkedIn](kedin.com/in/sanchit-rastogi-95964b314/)

⭐ *If you find this ML architecture useful for your research, consider starring the repository!*
