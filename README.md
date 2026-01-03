# 🚢 Titanic Survival Prediction: A TensorFlow Decision Forests Story

This repository contains my end-to-end solution for the [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic). My approach focuses on iterative improvement, feature engineering, and the "wisdom of the crowd" through ensemble modeling.

## 🎯 Project Goal
The goal is to predict passenger survival on the Titanic using demographics and trip data. Beyond just accuracy, this project explores **interpretability**—understanding *why* the model makes certain decisions.

## 🚀 Model Evolution & Results
I followed an iterative process to reach the final prediction:
* **Baseline Model**: Initial Gradient Boosted Trees (GBT) achieved **82.6%** accuracy.
* **Tuned Model**: Using a Random Search Tuner, I optimized hyperparameters to reach **91.8%** accuracy.
* **Final Ensemble**: To ensure stability, I trained an ensemble of **100 models** with different random seeds.
* **Final Output**: Predicted **150 survivors** out of 418 passengers in the test set.

## 🧬 Feature Engineering: "The Best Options"
The most significant breakthroughs came from creating new features that provided better social context:

### 1. Title Extraction 👑
I extracted titles (Mr., Mrs., Master, etc.) from passenger names and mapped them into categories.
* **Impact**: "Title" became the **#1 most important feature** in the model.
* **Insight**: It replaced the raw "Sex" feature in importance because it captured gender, age, and social status more effectively.

### 2. Family Size 👪
I combined `SibSp` (siblings/spouses) and `Parch` (parents/children) into a single `FamilySize` feature.
* **Formula**: `FamilySize = SibSp + Parch + 1`
* **Insight**: This allowed the model to see travel groups as a single unit, identifying how group size influenced survival chances.

## 📊 Feature Importance (Top 5)
1.  **Title** 👑 (Social context)
2.  **Fare** 💰 (Wealth indicator)
3.  **Name** 📛 (Specific family tokens)
4.  **Pclass** 🎫 (Socio-economic class)
5.  **Age** 🎂 (Vulnerability)

## 🛠️ Technology Stack
* **Language**: Python
* **Libraries**: TensorFlow Decision Forests (TF-DF), Pandas, NumPy, Matplotlib
* **Platform**: Kaggle Notebooks

---
*Created as part of a learning journey to master Gradient Boosted Trees and Feature Engineering.* 
