# 🚢 Titanic: Machine Learning from Disaster 🌊

## 🎯 The Goal
The objective of this project is to predict which passengers survived the Titanic shipwreck. Using a dataset of demographics and trip details, I built an ensemble machine learning model to uncover the social and economic factors that determined survival.

---

## 🚀 The Evolution of the Model
My approach followed a "Best Options Always" philosophy, iterating through three major phases:

1.  **Baseline GBT**: A default Gradient Boosted Trees model provided a solid accuracy of **82.6%**.
2.  **Hyperparameter Tuning**: By using a Random Search Tuner to optimize parameters like `max_depth` and `shrinkage`, the accuracy jumped to **91.8%**.
3.  **The Ensemble (Wisdom of the Crowd)**: I trained **100 different models** with varied random seeds and averaged their predictions. This smoothed out randomness and produced a stable survival count of **150 passengers**.

---

## 🧬 Feature Engineering (The Strategic Pivot)
The most significant gains came from transforming raw data into meaningful human contexts:

### 1. Title Extraction 👑
I parsed passenger names to extract social titles (e.g., *Mr., Mrs., Miss, Master, Rare*). 
* **The Impact**: This consolidated gender, age, and social status into one "super-feature". 
* **The Result**: Once **Title** was added, the model ranked it as the **#1 most important feature**. Interestingly, the importance of raw **Sex** dropped significantly because the Title already contained that information with better nuance.

### 2. FamilySize Calculation 👪
I combined the separate `SibSp` (siblings/spouses) and `Parch` (parents/children) columns into a single `FamilySize` feature ($FamilySize = SibSp + Parch + 1$).
* **The Logic**: Large families (size 5+) often struggled to stay together, while solo travelers (size 1) had different survival patterns.

---

## 📊 Results & Key Insights
The final Ensemble Model showed high confidence in its predictions, with clear probability peaks near 0 (likely deceased) and 1 (likely survived).

**Top 5 Features Driving Survival:**
1.  **Title** 👑 (Social/Gender context)
2.  **Fare** 💰 (Economic status)
3.  **Name** 📛 (Specific family ties)
4.  **Pclass** 🎫 (Ticket class)
5.  **Age** 🎂 (Physical vulnerability)

---

## 🛠️ How to Run
This project uses **TensorFlow Decision Forests (TF-DF)**. 
```python
import tensorflow_decision_forests as tfdf
# Load data, run preprocess, and fit the GBT Ensemble
model = tfdf.keras.GradientBoostedTreesModel(honest=True)
model.fit(train_ds)
