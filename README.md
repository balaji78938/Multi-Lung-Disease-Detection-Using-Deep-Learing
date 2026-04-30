# Multi-Lung-Disease-Detection-Using-Deep-Learing

# 🧠 Lung Disease Multi-Stage Classification Model 

---

## ⚙️ Environment & Setup

* **Device:** CUDA
* **Train Samples:** 6054
* **Validation Samples:** 2016

⚠️ Warning: Unauthenticated HuggingFace requests (set `HF_TOKEN`)

* **Total Parameters:** 20,998,896

---

## 🚀 Training Strategy

* **Epochs:** 25
* **Model Selection:**

```
Combined Score = 0.2 × Stage1 + 0.3 × Stage2 + 0.5 × Stage3
```

---

## 📊 Best Epoch Progression

| Epoch | S1 (%) | S2 (%) | S3 (%) | Combined (%) | Status  |
| ----- | ------ | ------ | ------ | ------------ | ------- |
| 1     | 96.53  | 97.46  | 67.83  | 82.46        | ⭐       |
| 2     | 96.88  | 99.38  | 77.81  | 88.09        | ⭐       |
| 3     | 97.02  | 99.07  | 81.92  | 90.09        | ⭐       |
| 5     | 98.02  | 99.69  | 81.67  | 90.35        | ⭐       |
| 6     | 95.63  | 99.44  | 83.29  | 90.61        | ⭐       |
| 8     | 96.48  | 99.57  | 83.29  | 90.81        | ⭐       |
| 10    | 96.88  | 99.26  | 84.54  | 91.42        | ⭐       |
| 11    | 98.16  | 99.07  | 85.29  | 92.00        | ⭐       |
| 12    | 97.47  | 99.07  | 86.28  | 92.36        | ⭐       |
| 13    | 98.21  | 99.26  | 87.16  | 93.00        | ⭐       |
| 18    | 98.16  | 99.63  | 87.28  | 93.16        | ⭐       |
| 21    | 98.66  | 99.75  | 87.66  | **93.49**    | 🏆 BEST |

---

## 🏆 Best Model

* **Epoch:** 21

* **Combined Score:** 93.49%

* **Stage 1 Accuracy:** 98.66%

* **Stage 2 Accuracy:** 99.75%

* **Stage 3 Accuracy:** 87.66%

---

## 📈 Final Evaluation

---

### 🩺 Stage 1: Normal vs Abnormal

* **Accuracy:** 98.66%

| Class    | Precision | Recall | F1-score | Support |
| -------- | --------- | ------ | -------- | ------- |
| Normal   | 0.95      | 0.98   | 0.97     | 402     |
| Abnormal | 1.00      | 0.99   | 0.99     | 1614    |

---

### 🦠 Stage 2: Disease Classification

* **Accuracy:** 99.75%

| Class     | Precision | Recall | F1-score | Support |
| --------- | --------- | ------ | -------- | ------- |
| COVID     | 0.99      | 1.00   | 1.00     | 406     |
| TB        | 1.00      | 1.00   | 1.00     | 406     |
| Pneumonia | 1.00      | 1.00   | 1.00     | 802     |

---

### 🫁 Stage 3: Pneumonia Type Classification

* **Accuracy:** 87.66%

| Class     | Precision | Recall | F1-score | Support |
| --------- | --------- | ------ | -------- | ------- |
| Bacterial | 0.88      | 0.88   | 0.88     | 401     |
| Viral     | 0.88      | 0.88   | 0.88     | 401     |

---

## 🏁 Final Results

* **Normal / Abnormal:** 98.66%
* **Disease Type:** 99.75%
* **Pneumonia Type:** 87.66%

---
