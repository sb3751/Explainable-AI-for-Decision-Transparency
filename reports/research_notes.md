## Local Explainability (SHAP vs LIME)

- SHAP explanations are consistent with global feature importance.
- LIME explanations vary more across similar instances.
- In some cases, LIME emphasizes locally influential but globally weak features.
- Disagreements highlight risks of relying on a single explanation method.

### Local Explanation Case Study (Instance 50)

For a representative high-risk instance, SHAP and LIME produced differing explanations.
SHAP attributed the prediction to multiple repayment-status features, reflecting the model’s
temporal reasoning and feature interactions. LIME, in contrast, highlighted fewer features
with stronger local influence, offering a simpler but less stable explanation.

This comparison suggests that while LIME improves human readability, it may oversimplify
the model’s decision logic, whereas SHAP provides more faithful but complex explanations.

# Research Questions & Hypotheses

## Research Motivation

Modern machine learning models often achieve higher predictive accuracy at the cost of transparency.
In high-stakes domains such as credit risk assessment, this trade-off raises concerns regarding trust,
fairness, and accountability. This project investigates whether explainable AI (XAI) techniques can
meaningfully bridge the gap between predictive performance and interpretability.

---

## Primary Research Question

**How does the trade-off between predictive performance and interpretability manifest in credit risk
models, and to what extent can explainable AI methods provide reliable decision insights for black-box
models?**

---

## Sub-Questions

1. **Performance Comparison**  
   How much predictive performance is gained when moving from interpretable models
   (Logistic Regression, Decision Trees) to black-box models (Random Forest, Gradient Boosting)?

2. **Global Decision Behavior**  
   Which feature groups dominate model decisions at a population level, and do black-box models rely
   primarily on behavioral credit information rather than demographic attributes?

3. **Local Decision Transparency**  
   Can individual predictions made by black-box models be meaningfully explained using post-hoc
   explainability techniques such as SHAP and LIME?

4. **Explanation Reliability**  
   How consistent and faithful are local explanations across different XAI methods, and where do
   disagreements arise?

---

## Working Hypothesis

Black-box models provide measurable improvements in predictive performance for credit risk prediction.
However, these gains come at the cost of direct interpretability. Explainable AI methods can partially
restore transparency by offering global and local insights into model behavior, but they do not fully
eliminate the inherent opacity of complex models.

---

## Scope and Limitations

- The study focuses on **post-hoc explainability**, not causal inference.
- Interpretability is evaluated in terms of **faithfulness and stability**, not legal compliance.
- Conclusions are specific to structured credit risk data and may not generalize to other domains.

# Results Interpretation & Insights

## 1. Performance vs Model Complexity

Across all experiments, predictive performance (measured primarily via ROC-AUC) increased
monotonically as model complexity increased from Logistic Regression to Gradient Boosting.
However, the magnitude of improvement was incremental rather than dramatic.

**Interpretation:**  
This suggests diminishing returns with increasing model complexity. While non-linear models
capture additional interactions, credit risk prediction remains constrained by data uncertainty
and class overlap. Higher complexity improves ranking quality but does not fundamentally
solve the problem.

---

## 2. Recall-Oriented Trade-offs in Credit Risk

Black-box models improved recall for the default class compared to interpretable baselines,
while overall accuracy remained relatively stable across models.

**Interpretation:**  
In credit risk settings, failing to identify defaulters is costlier than misclassifying safe clients.
The observed recall gains justify the use of more expressive models despite reduced transparency,
provided accountability mechanisms are in place.

---

## 3. Global Feature Dominance

Global SHAP analysis revealed that repayment behavior features consistently dominated model
decisions, while demographic features contributed marginally.

**Interpretation:**  
The dominance of behavioral credit history aligns with financial domain knowledge and suggests
that the model is not primarily relying on demographic proxies. This reduces concerns regarding
unintended socio-demographic bias at a population level.

---

## 4. Stability of Global Explanations Across Models

Feature importance rankings derived from SHAP were broadly consistent across Random Forest
and Gradient Boosting models.

**Interpretation:**  
Despite architectural differences, high-capacity models converge on similar decision signals.
This indicates that global explanation results reflect underlying data structure rather than
model-specific artifacts.

---

## 5. Local Explainability: SHAP vs LIME

Instance-level analysis showed that SHAP explanations were stable and reflected cumulative
feature contributions, whereas LIME explanations varied more and emphasized fewer features.

**Interpretation:**  
SHAP provides higher faithfulness to the model’s internal logic but at the cost of cognitive
complexity. LIME improves human readability but may oversimplify decision reasoning and
omit globally important features.

---

## 6. Limits of Explainability

Even with SHAP and LIME, explanations remain post-hoc and do not provide causal guarantees.

**Interpretation:**  
Explainable AI methods mitigate opacity but cannot fully restore transparency. Responsible
deployment of black-box models requires combining explainability tools with human oversight
and domain expertise.

# Explainable AI for Decision Transparency in Credit Risk

## Abstract

High-performing machine learning models for credit risk prediction often lack transparency,
raising concerns about trust and accountability in high-stakes decision-making. This study
investigates the trade-off between predictive performance and interpretability by comparing
interpretable baseline models with black-box ensemble methods and evaluating whether
post-hoc explainable AI (XAI) techniques can provide meaningful decision insights.

---

## 1. Introduction

Credit risk assessment is a critical financial task where incorrect decisions can have
significant economic and social consequences. While complex machine learning models
offer improved predictive performance, their opaque decision logic poses challenges for
regulatory compliance and user trust.

This work examines whether explainability methods such as SHAP and LIME can mitigate
the loss of transparency introduced by black-box models, without negating performance
benefits.

---

## 2. Research Questions

This study addresses the following questions:

1. How does predictive performance change as models move from interpretable to black-box?
2. Which features dominate decisions at a population level?
3. Can individual predictions be meaningfully explained?
4. How reliable are explanations across different XAI methods?

---

## 3. Methodology

### Dataset
The UCI Credit Card Default dataset (Taiwan) was used, consisting of 30,000 samples and
23 numeric features related to credit limits, repayment history, billing amounts, and
demographic attributes.

### Models
- Interpretable baselines: Logistic Regression, Decision Tree
- Black-box models: Random Forest, Gradient Boosting

### Explainability
- Global explainability: SHAP summary plots
- Local explainability: SHAP waterfall plots and LIME local explanations

---

## 4. Results

Predictive performance increased monotonically with model complexity, with Gradient
Boosting achieving the highest ROC-AUC. However, accuracy gains were incremental,
indicating diminishing returns.

Global SHAP analysis showed that repayment behavior features dominated model decisions,
while demographic features had limited influence. Local analysis revealed that SHAP
provided stable, faithful explanations, whereas LIME produced simpler but more variable
local explanations.

---

## 5. Discussion

The results highlight a clear trade-off between interpretability and performance. While
black-box models improve default detection, they sacrifice direct transparency. Explainability
methods partially bridge this gap but remain post-hoc approximations without causal guarantees.

A hybrid deployment strategy combining high-performance models, explainability tools, and
human oversight appears most appropriate for credit risk applications.

---

## 6. Conclusion

This study demonstrates that explainable AI can enhance transparency in black-box credit
risk models but cannot fully replace inherently interpretable approaches. Responsible use of
complex models requires both technical explainability and informed human judgment.

---

## Keywords
Explainable AI, Credit Risk, SHAP, LIME, Model Interpretability, Trustworthy AI
