# Thiranex_predictive_modelling
This repository demonstrates supervised learning techniques to build predictive models using Python. The project applies algorithms such as Linear Regression, Decision Trees, and Random Forests to forecast outcomes based on structured datasets.
#  Car Evaluation Classification using Machine Learning

##  Overview
This project applies **supervised machine learning** techniques to classify car evaluations into categories (`acc`, `good`, `unacc`, `vgood`).  
We experiment with **Logistic Regression, Decision Tree, and Random Forest**, compare their performance, and visualize results using **confusion matrices** and **ROC curves**.

---

##  Dataset
- **File:** `car_evaluation(updated).csv`  
- **Target column:** `decision`  
- **Classes:** `acc`, `good`, `unacc`, `vgood`  
- **Features:** Buying price, maintenance, doors, persons, lug_boot, safety (categorical, encoded).  

---

##  Workflow
1. **Data Preprocessing**
   - Label encoding for categorical features  
   - Feature scaling using `StandardScaler`  
   - Train-test split (80/20)  

2. **Model Training**
   - Logistic Regression  
   - Decision Tree  
   - Random Forest  

3. **Evaluation**
   - Accuracy, Precision, Recall, F1-score  
   - Confusion Matrix heatmaps  
   - Multi-class ROC curves (Random Forest)  

---

## Results

###  Logistic Regression
- Accuracy: **66.18%**  
- Performs well on `unacc` class but poorly on `good` and `vgood`.

###  Decision Tree
- Accuracy: **97.40%**  
- Strong performance across all classes, especially `unacc`.

###  Random Forest
- Accuracy: **97.69%**  
- Best overall performance, balanced precision/recall across all classes.

---

##  Model Comparison

| Model               | Accuracy |
|----------------------|----------|
| Logistic Regression  | 0.6618   |
| Decision Tree        | 0.9740   |
| Random Forest        | 0.9769   |

---

## Visualizations
The following plots were generated during evaluation:
<img width="1165" height="523" alt="result" src="https://github.com/user-attachments/assets/8d767f2d-0907-406f-89fc-cea2d129f514" />


- **Confusion Matrices**  
  - Logistic Regression → shows misclassifications in `good` and `vgood`.  
  - Decision Tree → near-perfect diagonal dominance.  
  - Random Forest → excellent accuracy across all classes.  

- **ROC Curves (Random Forest)**  
  - Multi-class ROC curves with AUC values:  
    - `acc` → 0.998  
    - `good` → 1.000  
    - `unacc` → 1.000  
    - `vgood` → 1.000  

These visualizations confirm that **Random Forest** is the most robust model for this dataset.

---

##  Expected Outcome
- Hands-on experience with **classification algorithms**.  
- Understanding of **model evaluation metrics**.  
- Ability to compare models and select the best one.  
- Visual storytelling with confusion matrices and ROC curves.  

---

##  How to Run
1. Clone the repository.  
2. Place `car_evaluation(updated).csv` in the project folder.  
3. Run the script:  
   ```bash
   python thiranex_predictive_modelling_code.py

Trying other ensemble methods (XGBoost, LightGBM)
