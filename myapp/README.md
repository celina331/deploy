# 🚢 Titanic Survival Prediction Project

##  Project Overview

During the semester, we learned many concepts in **Data Analysis, Data Science, and Machine Learning**.  
This project summarizes and applies these concepts using the **Titanic dataset**.

The objective is to:
- Analyze passenger data
- Perform data preprocessing
- Visualize important patterns
- Apply a Machine Learning model (Linear Regression)
- Predict passenger survival

---

##  Dataset Information

**Dataset:** Titanic Dataset  

The dataset contains the following features:
- PassengerId  
- Survived  
- Pclass  
- Name  
- Sex  
- Age  
- SibSp  
- Parch  
- Ticket  
- Fare  
- Cabin  
- Embarked  

###  Target Variable
- **Survived**
  - `0` → Passenger did not survive  
  - `1` → Passenger survived  

---

## 🛠 Technologies Used

- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Streamlit  
- Scikit-learn  

---

##  Applied Labs During the Semester

###  Lab 1 — Statistical Foundations in Data Analysis
We performed basic statistical analysis on the dataset:
- Mean
- Median
- Standard deviation
- Minimum and maximum values  

We used:
```python
df.describe()

###  Lab 2 — Inferential Statistics

We analyzed relationships between variables using:

- Correlation analysis  
- Probability interpretation  
- Survival statistics  

**Example:**

df.corr(numeric_only=True)
