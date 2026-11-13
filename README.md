# 🛒 Will This Visitor Buy? Real-Time Purchase Intent

**Most online-shop visitors leave without buying; this app predicts from a browsing session who is likely to purchase, so a site can show a coupon or chat only where it helps.**

🔗 **Live app:** [ADD YOUR STREAMLIT LINK]
📸 *[Add a screenshot or GIF of the app here]*

## Results (test set, never used for tuning)

| Model | ROC-AUC | PR-AUC | Recall | Precision | F1 |
|---|---|---|---|---|---|
| Baseline (majority class) | 0.500 | 0.157 | 0.000 | 0.000 | 0.000 |
| Logistic Regression | 0.927 | 0.696 | 0.798 | 0.595 | 0.682 |
| Random Forest | 0.933 | 0.745 | 0.715 | 0.649 | 0.680 |
| **Gradient Boosting (final)** | **0.937** | **0.748** | **0.728** | **0.662** | **0.693** |

*Accuracy is not the headline metric: a model that never predicts a purchase already scores 84.4%, because only ~15.7% of sessions end in a purchase.*

## What I did
- EDA on ~12K real shopping sessions; engineered pages/duration features inside a scikit-learn `Pipeline` (split before fitting, no leakage).
- Compared a baseline with Logistic Regression, Random Forest and Gradient Boosting using 5-fold CV + grid search on the training set only.
- Tuned the decision threshold on out-of-fold training predictions, then evaluated once on the test set (confusion matrix, ROC/PR curves, error analysis, permutation importance).
- Turned probabilities into an action policy: no action / coupon / chat.

## Run it yourself
```bash
pip install -r requirements.txt
# 1) run Shopper_Purchase_Intent.ipynb top to bottom (creates purchase_intent_model.joblib)
# 2) start the app
streamlit run app.py
```
Deploy: push `app.py`, `features.py`, `requirements.txt` and `purchase_intent_model.joblib` to GitHub, then create an app on [Streamlit Community Cloud](https://streamlit.io/cloud).

## Limitations
Single shop, one year of data; the model shows correlation, not whether a coupon *causes* a sale; the action thresholds are business assumptions that should be A/B tested.

## Data
Online Shoppers Purchasing Intention Dataset, UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/468
