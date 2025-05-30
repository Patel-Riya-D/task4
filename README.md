# 🧠 Logistic Regression for Breast Cancer Detection

This project builds a **binary classification model** using **Logistic Regression** to classify tumors as malignant or benign using the Breast Cancer Wisconsin dataset.

---

## 📂 Dataset

* Source: [Kaggle – Breast Cancer Wisconsin (Diagnostic)](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
* File used: `data.csv`

---

## 🛠️ Libraries Used

* numpy
* pandas
* seaborn
* matplotlib
* scikit-learn

---

## 🚀 Main Functions & Their Purpose

### 1. `train_test_split()`

```python
X_train, X_test, y_train, y_test = train_test_split(...)
```

* **Purpose**: Splits the dataset into training and testing sets (typically 80/20 split).

---

### 2. `StandardScaler()`

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

* **Purpose**: Normalizes the data to improve model performance.

---

### 3. `LogisticRegression().fit()`

```python
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
```

* **Purpose**: Trains the logistic regression model on the scaled training data.

---

### 4. `predict()` and `predict_proba()`

```python
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]
```

* **Purpose**: Generates predicted classes (`predict`) and predicted probabilities (`predict_proba`).

---

### 5. `confusion_matrix()`

```python
confusion_matrix(y_test, y_pred)
```

![image](https://github.com/user-attachments/assets/924aea15-8199-48cc-8157-f63ed1961193)


* **Purpose**: Evaluates model performance by comparing predictions vs. actual values.

* **Confusion Matrix Output**:

```
              Predicted
              0     1
Actual  0    85     5
        1    10    50
```

---

### 6. `classification_report()`

```python
classification_report(y_test, y_pred)
```

* **Purpose**: Provides precision, recall, F1-score, and accuracy.

---

### 7. `roc_curve()` and `roc_auc_score()`

```python
fpr, tpr, threshold = roc_curve(y_test, y_prob)
roc_auc_score(y_test, y_prob)
```

* **Purpose**: Evaluates how well the model distinguishes between classes using the ROC-AUC metric.

* **ROC Curve Sample Points**:
  \| False Positive Rate (FPR) | True Positive Rate (TPR) |
  \|---------------------------|---------------------------|
  \| 0.00                      | 0.00                      |
  \| 0.00                      | 0.02                      |
  \| 0.00                      | 1.00                      |
  \| 1.00                      | 1.00                      |

* **AUC**: 1.00

---

### 8. `sigmoid()`

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

* **Purpose**: Used to map predictions to probabilities between 0 and 1 in logistic regression.

---

