# Case Studies — Data Science Projects Lab

> Personal data science projects built outside coursework: classification, neural networks, time series forecasting, conversational AI, and business intelligence visualization.

---

## Table of Contents

1. [Logistic Regression — Breast Cancer Classification](#1-logistic-regression--breast-cancer-classification)
2. [ANN — Power Plant Energy Output Prediction](#2-ann--power-plant-energy-output-prediction)
3. [Classical Time Series Forecasting Reference](#3-classical-time-series-forecasting-reference)
4. [ChatBot Jordan — Identity-Verified Conversational AI](#4-chatbot-jordan--identity-verified-conversational-ai)
5. [Tableau BI Dashboards](#5-tableau-bi-dashboards)

---

## 1. Logistic Regression — Breast Cancer Classification

**Domain:** Binary Classification / Medical ML
**Stack:** Python, scikit-learn, pandas
**Dataset:** UCI Breast Cancer (Wisconsin), 683 samples, 10 features

### Problem
Classify breast tissue samples as benign (class 2) or malignant (class 4) based on clinical measurement features. A medically grounded binary classification problem with direct real-world stakes: false negatives (missing malignant cases) are more costly than false positives.

### Approach
Applied logistic regression with L2 regularization (`lbfgs` solver, `C=1.0`) to the UCI Breast Cancer dataset.

Data split: 80% training (546 samples), 20% test (137 samples). Evaluated performance via confusion matrix and 50-fold cross-validation to ensure the result wasn't a lucky split artifact.

**Features:** Clump thickness, uniformity of cell size, uniformity of cell shape, marginal adhesion, single epithelial cell size, bare nuclei, bland chromatin, normal nucleoli, mitoses.

### Results
| Metric | Value |
|--------|-------|
| Test Accuracy | **95.62%** |
| 50-Fold CV Mean Accuracy | **97.05%** |
| CV Standard Deviation | 0.047 |
| Confusion Matrix | [[83, 3], [3, 48]] |
| Total Misclassifications | 6 / 137 |

The 50-fold cross-validation mean (97.05%) outperforming the single test split (95.62%) confirms the model generalizes well and the test split wasn't unusually favorable.

### Takeaways
This project demonstrates the disciplined ML workflow: train/test split → evaluate → validate with cross-validation. The confusion matrix framing matters here — 3 false negatives (missed malignant cases) vs. 3 false positives (benign flagged as malignant) has asymmetric consequences, which informs threshold decisions in production medical systems.

---

## 2. ANN — Power Plant Energy Output Prediction

**Domain:** Regression / Deep Learning
**Stack:** Python, TensorFlow 2.4.1, Keras, pandas
**Dataset:** Combined Cycle Power Plant (UCI), 9,568 samples, 4 features

### Problem
Predict the net hourly electrical energy output (PE) of a combined cycle power plant based on ambient environmental conditions. A regression problem where accurate prediction directly impacts grid planning and energy dispatch decisions.

### Approach
Built a feedforward neural network from scratch using TensorFlow/Keras.

**Architecture:**
```
Input Layer:  4 features (Temperature, Ambient Pressure, Relative Humidity, Exhaust Vacuum)
Hidden Layer 1: 6 units, ReLU activation
Hidden Layer 2: 6 units, ReLU activation
Output Layer:  1 neuron, linear (regression output)
```

**Training configuration:**
- 500 epochs, batch size 32
- Adam optimizer, MSE loss function
- 80/20 train-test split (~7,654 train, ~1,914 test)

### Results
- **Loss converged to ~25 MSE around epoch 30** — the remaining 470 epochs yielded negligible improvement
- This demonstrates a key production insight: early stopping would preserve training compute without sacrificing accuracy
- Loss curve showed rapid descent in early epochs, followed by stable plateau — characteristic of a well-specified architecture for this problem size

### Takeaways
The most practically useful finding is the convergence behavior: the network effectively learned by epoch 30, meaning the 500-epoch run was overlong by design (for observability). In a production pipeline, this would inform an early stopping callback with patience ~10–15 epochs. The project demonstrates both the model-building skill and the ability to interpret training dynamics critically.

---

## 3. Classical Time Series Forecasting Reference

**Domain:** Time Series Analysis / Forecasting
**Stack:** Python, statsmodels, pandas, matplotlib
**Dataset:** Daily minimum temperatures in Melbourne (multivariate time series)

### Problem
Time series problems span a wide family of models — from simple autoregression to full multivariate exponential smoothing systems. The challenge for practitioners: knowing which method fits which data structure. This notebook was built as a personal reference to implement and compare all classical methods in one place.

### Approach
Implemented 11 forecasting methods sequentially, each with:
- **Stationarity testing** (ADF test) — determining if differencing is required
- **ACF/PACF analysis** — identifying lag structures for AR and MA terms
- **Model fit** — implemented via `statsmodels`
- **Residual diagnostics** — checking for autocorrelation in residuals post-fit

**Methods implemented:**
| Method | Type | Use Case |
|--------|------|----------|
| AR | Univariate | Autoregression on lagged values |
| MA | Univariate | Moving average on lagged errors |
| ARMA | Univariate | Combined AR + MA |
| ARIMA | Univariate | ARMA with differencing for non-stationarity |
| SARIMA | Univariate | ARIMA with seasonal component |
| SARIMAX | Univariate | SARIMA with exogenous variables |
| VAR | Multivariate | Vector autoregression across multiple series |
| VARMA | Multivariate | VAR with moving average terms |
| VARMAX | Multivariate | VARMA with exogenous variables |
| SES | Exponential | Simple exponential smoothing |
| HWES | Exponential | Holt-Winters with trend + seasonality |

### Results
- SES prediction: ~99.3; HWES prediction: ~100 (on Melbourne temperature holdout)
- Full model-selection decision tree embedded in notebook structure — model selection flows from data properties (stationarity, seasonality, multivariate structure)

### Takeaways
The value of this notebook is not any single model result — it's the systematic coverage. Being able to move fluently between AR, ARIMA, and Holt-Winters based on data characteristics is a core forecasting competency for BI, supply chain, and financial analytics roles. This notebook serves as a reusable implementation reference.

---

## 4. ChatBot Jordan — Identity-Verified Conversational AI

**Domain:** Conversational AI / NLP
**Stack:** Python, ChatterBot, SQLite
**Framework:** ChatterBot 1.x with BestMatch logic adapter

### Problem
Build a chatbot that can handle natural language conversation while also enforcing identity-based access control — a capability needed in secure or personalized contexts (customer service bots with tiered access, internal tools, home automation interfaces).

### Approach
Designed a two-layer system:
1. **Identity layer** — input parsing to detect user identity (Mark or Jane) before the bot responds
2. **Conversation layer** — ChatterBot with BestMatch logic adapter, trained on:
   - ChatterBot's built-in English language corpus
   - Custom conversation pairs via `ListTrainer` (personalized responses per user)

**Architecture:**
```
main.py        ← User interaction loop + identity check
functions.py   ← Bot factory (create_bot, train_all_data, custom_train, start_chatbot)
SQLite DB      ← Persistent conversation storage (read_only=False for continuous learning)
```

The modular design separates bot instantiation, training, and interaction — each in its own function — making it extensible without rewriting the interaction loop.

### Results
- Successfully implemented identity-gated conversation flow
- Bot improves from each session — conversations are stored and weighted in future responses
- Custom training pairs override corpus defaults for user-specific responses

### Takeaways
The identity layer on top of a corpus-trained chatbot demonstrates compositional system design: standard NLP capabilities plus a custom access control layer, without modifying the underlying model. This pattern — wrapping an ML component with business logic — is directly applicable to enterprise chatbot and agent deployment.

---

## 5. Tableau BI Dashboards

**Domain:** Business Intelligence / Data Visualization
**Stack:** Tableau Desktop, Tableau Public

### Problem
Build analytical dashboards that move beyond raw data — designing views that help users answer specific business questions quickly, without requiring them to query or filter data themselves.

### Dashboards Built

**AdTech Full Funnel Dashboard**
A complete marketing funnel visualization from impression through conversion. Designed to surface drop-off rates at each funnel stage, segment performance, and campaign-level ROI signals. Built to the full-funnel analytical pattern used by performance marketing teams.

**Data Detective — Exploratory Dashboard**
An exploration-oriented dashboard designed to support open-ended data investigation. Filters, cross-highlighting, and drill-down designed to let analysts move between hypotheses quickly.

**Product Analytics Templates**
Practice workbooks covering: dimension filtering patterns, discrete vs. continuous date aggregation, product view funnels, and sorting by fields not directly in the view — core techniques for product analytics reporting.

### Takeaways
Tableau dashboards are only as useful as their ability to answer the right question without requiring the viewer to know SQL. These projects demonstrate awareness of the analyst-to-stakeholder design problem: what decisions does this person need to make, and what view supports that decision fastest?

→ [View on Tableau Public](https://public.tableau.com/profile/sidd.chauhan)
