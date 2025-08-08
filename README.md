# 💳 Credit Card Fraud Detection using ML & SHAP

A complete fraud detection pipeline built using machine learning algorithms like Decision Tree, Random Forest, and XGBoost — with class imbalance handled via SMOTE and model interpretability powered by SHAP. This project tackles the real-world challenge of identifying fraudulent credit card transactions in highly imbalanced data.

---

## 📌 Overview

Credit card fraud is a major concern in digital finance, where fraudulent transactions are rare but extremely costly. This project demonstrates an end-to-end approach to detect such anomalies using interpretable machine learning models, handling imbalance, and drawing insights from the data.

---

## 🧠 Key Concepts Covered

- Machine Learning for fraud detection
- Handling imbalanced datasets with **SMOTE**
- Model performance evaluation: **Confusion Matrix**, **Precision**, **Recall**, **F1-Score**, **ROC Curve**
- Interpretable AI using **SHAP** (SHapley Additive Explanations)

---

## 🚀 Implementation Highlights

- ✅ Trained baseline models: **Decision Tree** and **Random Forest**  
- ✅ Applied **SMOTE** to balance fraud and non-fraud transactions  
- ✅ Built advanced model using **XGBoost** with tuned hyperparameters  
- ✅ Evaluated model with ROC-AUC, confusion matrix, and F1-score  
- ✅ Applied **SHAP** to visualize and interpret model predictions

---

## 📈 Model Insights

- **Before SMOTE**: High precision, but low recall for fraudulent class  
- **After SMOTE**: Improved recall with balanced F1-score  
- **Top Predictive Features (via SHAP)**: `V12`, `Amount`, `V17`, `V14`

Visualizations include:
- SHAP Summary Plot  
- SHAP Beeswarm Plot  
- SHAP Force & Waterfall Plots  
- ROC Curve  
- Confusion Matrix  

---

## 🧭 What We Learned

- Gained a strong understanding of various fraud types:  
  _skimming, phishing, counterfeit cards, merchant collusion, site cloning, triangulation_
- Studied secure payment technologies like **AVS**, **CVV**, **EMV**, **3D Secure**, and **SET protocol**
- Understood why fraud detection systems must be both accurate and interpretable in real-world applications

---

## 🔮 Future Scope

- ⚡ **Real-Time Fraud Detection** using Kafka + Spark streaming  
- 🧠 **Temporal Modeling** using LSTM for sequential transaction analysis  
- 🧩 **Advanced Explainability** using LIME and counterfactual explanations  
- 🏦 **Banking Integration** with real-time APIs and dashboards  


---

## 📫 Contact

Feel free to open an issue or discussion in the GitHub repository if you have questions or suggestions!

