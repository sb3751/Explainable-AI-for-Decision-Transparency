Explainable AI for Decision Transparency
Credit Default Prediction using SHAP and LIME
📌 Overview

This project studies why machine learning models make certain predictions, focusing on decision transparency in credit risk modeling. While complex models improve predictive performance, they often operate as black boxes. This work evaluates whether explainability techniques can meaningfully bridge the gap between accuracy and interpretability.

The project is structured as a research-oriented ML pipeline, progressing from interpretable baselines to black-box models and finally to global and local explainability analysis.

🎯 Objectives

Build predictive models for credit default risk

Compare interpretable vs black-box models

Analyze performance vs transparency trade-offs

Apply SHAP (global & local) and LIME (local)

Evaluate faithfulness, stability, and limitations of explanations

🧠 Research Questions

Do black-box models significantly outperform interpretable models?

Which features dominate decisions at a population level?

Can individual predictions be meaningfully explained?

Where do SHAP and LIME agree — and where do they diverge?

Can explainability tools fully restore trust in opaque models?

📂 Project Structure
explainable-ai-credit-risk/
│
├── data/
│   └── raw/
│       └── credit_default.csv
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── split.py
│   ├── baselines.py
│   ├── blackbox.py
│   ├── explain_shap.py
│   ├── explain_lime.py
│   └── model_wrappers.py
│
├── experiments/
│   ├── run_data_check.py
│   ├── run_baselines.py
│   ├── run_blackbox.py
│   ├── run_global_explainability.py
│   └── run_local_explainability.py
│
├── reports/
│   ├── figures/
│   └── research_notes.md
│
├── requirements.txt
└── README.md

📊 Dataset

Source: UCI Credit Card Default Dataset (Taiwan)

Samples: 30,000

Features: 23 numeric attributes

Target: Default payment next month (Y)

Task: Binary classification

Class imbalance: ~78% non-default, ~22% default

The dataset was validated and cleaned to remove header artifacts and ensure numeric consistency.

🧪 Methodology
1️⃣ Interpretable Baselines

Logistic Regression

Shallow Decision Tree

Purpose:

Establish transparent performance benchmarks before introducing complexity.

2️⃣ Black-Box Models

Random Forest

Gradient Boosting (best performer)

Purpose:

Capture non-linear interactions and improve default detection.

3️⃣ Global Explainability (SHAP)

SHAP summary plots on test data

Feature dominance analysis

Hypothesis validation

4️⃣ Local Explainability (SHAP vs LIME)

Instance-level SHAP waterfall plots

LIME local surrogate explanations

Direct comparison on identical instances

A wrapper was implemented to preserve feature name consistency during LIME explanations.

📈 Results Summary
Model	ROC-AUC	Recall (Default)	Interpretability
Logistic Regression	~0.71	Low	High
Decision Tree	~0.73	Medium	Medium-High
Random Forest	~0.75	Medium	Low
Gradient Boosting	~0.78	Medium-High	Lowest

Key finding:
Performance improves with complexity, but transparency decreases sharply.

🔍 Explainability Insights
Global (SHAP)

Repayment history dominates predictions

Behavioral features outweigh demographics

Model reasoning aligns with domain intuition

Local (SHAP vs LIME)

SHAP is more faithful and stable

LIME is simpler but more variable

Explanations can be plausible without being faithful

Explainability mitigates opacity but does not eliminate it.

⚖️ Interpretability vs Performance Trade-off

Black-box models offer meaningful performance gains only when accompanied by explainability analysis.

Recommended deployment strategy:

High-performance model

Global monitoring with SHAP

Local accountability via SHAP + LIME

Human review for high-impact decisions

▶️ How to Run
Requirements

Python 3.9+

Base Python environment (no virtualenv required)

Install dependencies
pip install -r requirements.txt

Verify dataset
python -m experiments.run_data_check

Train baseline models
python -m experiments.run_baselines

Train black-box models
python -m experiments.run_blackbox

Global explainability
python -m experiments.run_global_explainability

Local explainability
python -m experiments.run_local_explainability


Outputs are saved to:

reports/figures/

🔮 Extensions & Future Work

Bias & fairness analysis (demographic parity, equalized odds)

Counterfactual explanations

Threshold optimization for recall-critical use cases

Temporal sequence modeling

Regulatory-focused explanation reporting

🎓 Academic Positioning

This project demonstrates:

Research-first ML development

Awareness beyond accuracy metrics

Responsible AI thinking

Explainability limitations, not just strengths

It aligns strongly with M.Tech (CSE) research expectations, especially in Japan-style academic evaluation.
