# Titanic: Machine Learning from Disaster 🌊

## 🎯 The Goal
The objective of this project is to predict which passengers survived the Titanic shipwreck. 
Using the classic Titanic dataset, we aimed to build a model that captures the underlying 
patterns of survival based on passenger demographics, wealth, and social standing.
🧬 Section 2: Feature Engineering (The Strategic Pivot)
This is where you explain how you transformed raw data into meaningful "clues" for the model.

## 🛠️ Feature Engineering: Finding the Best Options
To improve our predictions, we moved beyond the basic features provided in the dataset. 
We focused on two key transformations:

* **Title Extraction 👑**: We parsed passenger names to extract titles (e.g., Mr., Mrs., Master). 
  This consolidated information about gender, age, and social status into a single, high-impact feature.
* **FamilySize Calculation 👪**: We combined `SibSp` (siblings/spouses) and `Parch` (parents/children) 
  into a `FamilySize` feature ($FamilySize = SibSp + Parch + 1$). This helped the model understand 
  the total group size of each passenger.
