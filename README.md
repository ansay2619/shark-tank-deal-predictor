# 🦈 Shark Tank Deal Predictor

An end-to-end Machine Learning application that predicts whether an entrepreneur will receive an investment offer on Shark Tank based on their pitch metrics.

🔗 **Live Demo:** [Shark Tank Deal Predictor App](https://shark-tank-deal-predictor-9xeyymqmjhbh6mqdrl5tsx.streamlit.app/)

---

## 📌 Project Overview
- **Objective:** Predict investment deal outcomes (`Got Deal`: 1 / 0) using pre-pitch startup attributes while eliminating post-pitch data leakage.
- **Models Evaluated:** Logistic Regression, Decision Tree, Random Forest, XGBoost.
- **Top Model:** Logistic Regression (achieving high recall of ~0.91, composite rank #1).

---

## 🛠️ Pipeline Architecture
1. **Data Preprocessing:**
   - Median imputation & `StandardScaler` for numeric metrics (Ask Amount, Offered Equity, Valuation Requested, US Viewership).
   - Most-frequent imputation & `OneHotEncoder` for categorical metrics (Industry, Pitchers Gender).
2. **Artifact Serialization:** Exported preprocessing pipeline and model via `joblib`.
3. **Deployment:** Hosted interactive web application on Streamlit Community Cloud.

---

## 💻 Local Setup
```bash
git clone [https://github.com/ansay2619/shark-tank-deal-predictor.git](https://github.com/ansay2619/shark-tank-deal-predictor.git)
cd shark-tank-deal-predictor
pip install -r requirements.txt
streamlit run app.py
