# Data Science Projects Lab

> Personal implementations built outside coursework — covering neural networks, classification, time series forecasting, conversational AI, and data visualization.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)
![Tableau](https://img.shields.io/badge/Tableau-E97627?style=flat&logo=tableau&logoColor=white)

---

## Highlights

- **97.05% classification accuracy** — Logistic Regression on Breast Cancer UCI dataset (50-fold CV)
- **11 forecasting methods** in a single notebook — AR through Holt-Winters, with stationarity testing and residual diagnostics
- **Full-stack chatbot** — identity-verified access control, corpus training, and persistent learning via SQLite

---

## Projects at a Glance

| Project | Type | Dataset | Key Result |
|---------|------|---------|------------|
| [Logistic Regression](./Logistic_Regression_Practical_Example.ipynb) | Classification | Breast Cancer UCI (683 samples) | 95.62% test accuracy, 97.05% CV |
| [ANN Price Prediction](./ANN_Example_SC.ipynb) | Regression (Neural Net) | Combined Cycle Power Plant (9,568 samples) | Loss converged ~25 MSE by epoch 30 |
| [Classical Time Series](./Classical_Time_Series_Forecasting.ipynb) | Forecasting | Daily min temperatures (Melbourne) | 11 methods: AR, MA, ARIMA, SARIMA, HWES, + more |
| [ChatBot Jordan](./Chatbot) | Conversational AI | Custom + ChatterBot English corpus | Identity-gated, self-improving chatbot |
| [Tableau Dashboards](./Tableau) | Data Visualization | AdTech, product analytics | Full funnel dashboard + product view templates |

---

## Project Details

### Logistic Regression — Breast Cancer Classification
`Logistic_Regression_Practical_Example.ipynb`

Binary classification on the UCI Breast Cancer dataset (683 samples, 10 features). Scikit-learn logistic regression with L2 regularization, evaluated via confusion matrix and 50-fold cross-validation.

- **Test accuracy: 95.62%**
- **50-fold CV mean accuracy: 97.05%** (std: 0.047) — confirms robust generalization
- Confusion matrix: [[83, 3], [3, 48]] — 6 total misclassifications on 137-sample test set

→ [View notebook](./Logistic_Regression_Practical_Example.ipynb)

---

### ANN — Power Plant Energy Output Prediction
`ANN_Example_SC.ipynb`

Regression neural network built with TensorFlow/Keras to predict energy output of a Combined Cycle Power Plant. Demonstrates end-to-end deep learning pipeline: architecture design, training, and loss convergence analysis.

- **Dataset:** 9,568 samples, 4 features (Temperature, Ambient Pressure, Relative Humidity, Exhaust Vacuum)
- **Architecture:** Input(4) → Dense(6, ReLU) → Dense(6, ReLU) → Output(1)
- **Training:** 500 epochs, Adam optimizer, MSE loss, 80/20 train-test split
- **Result:** Loss converged to ~25 MSE around epoch 30 — demonstrates early stopping as a practical optimization

→ [View notebook](./ANN_Example_SC.ipynb)

---

### Classical Time Series Forecasting Reference
`Classical_Time_Series_Forecasting.ipynb`

Comprehensive reference notebook implementing 11 classical forecasting methods on Melbourne's daily minimum temperature dataset. Each method is accompanied by stationarity testing (ADF), ACF/PACF analysis, and residual diagnostics.

**Methods covered:**
AR · MA · ARMA · ARIMA · SARIMA · SARIMAX · VAR · VARMA · VARMAX · Simple Exponential Smoothing (SES) · Holt-Winters Exponential Smoothing (HWES)

Built as a practical reference — any time series problem can be mapped to one of these methods using the decision flow built into the notebook.

→ [View notebook](./Classical_Time_Series_Forecasting.ipynb)

---

### ChatBot Jordan
`Chatbot/`

Python chatbot using ChatterBot with identity-based access control. Built to demonstrate: corpus-based NLP training, custom knowledge base injection, and persistent self-improvement via SQLite storage.

- **Identity verification:** Access control gating for two users (Mark and Jane)
- **Training:** ChatterBot English corpus + custom conversation pairs via ListTrainer
- **Storage:** SQLite backend with `read_only=False` — the bot improves from every conversation
- **Architecture:** BestMatch logic adapter with modular `create_bot()`, `train_all_data()`, and `custom_train()` functions

→ [View code](./Chatbot/)

---

### Tableau Dashboards
`Tableau/`

Collection of Tableau workbooks including an AdTech Full Funnel Dashboard and product analytics templates. Built to practice BI visualization patterns for funnel analysis, dimension filtering, and cross-date aggregation.

→ [View on Tableau Public](https://public.tableau.com/profile/sidd.chauhan)

---

→ See full methodology breakdowns and outcome summaries in [`CASE_STUDY.md`](./CASE_STUDY.md)
