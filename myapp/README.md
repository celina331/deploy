#  Titanic Survival Prediction Project

##  Project Overview

During the semester, we learned many concepts in **Data Analysis, Data Science, and Machine Learning**.  
This project summarizes and applies these concepts using the **Titanic dataset**.

The objective is to:
- Analyze passenger data
- Perform data preprocessing
- Visualize important patterns
- Apply a Machine Learning model (Linear Regression)
- Predict passenger survival



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



##  Technologies Used

- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Streamlit  
- Scikit-learn  



##  Applied Labs During the Semester

###  Lab 1 — Statistical Foundations in Data Analysis
We performed basic statistical analysis on the dataset:
- Mean
- Median
- Standard deviation
- Minimum and maximum values  

```python
df.describe()
```

### Lab 2 — Inferential Statistics

We analyzed relationships between variables using:

- Correlation analysis
- Probability interpretation
- Survival statistics

**Example:**

```python
df.corr(numeric_only=True)
```
###  Lab 3 — Dataset Loading

The Titanic dataset was loaded and explored using Pandas to better understand its structure and content before preprocessing and analysis.

###  Operations Performed

- Loading the CSV dataset into a DataFrame  
- Previewing the first rows of the dataset  
- Analyzing dataset structure and column types  
- Generating statistical summaries for numerical features  

The following functions were used during exploration:

- `df.head()` → displays the first rows of the dataset  
- `df.info()` → provides information about columns and missing values  
- `df.describe()` → generates statistical summaries  

###  Example

write this exactly for the readme file  ```python

df = pd.read_csv("titanic.csv")

###  Lab 4 — Data Preprocessing

Before applying analysis and machine learning techniques, the dataset required preprocessing to improve data quality and model performance.

###  Steps Performed

- Handling missing values in important columns such as `Age` and `Fare`

- Removing irrelevant columns that were not useful for prediction:

  - `Cabin`

  - `Name`

  - `Ticket`

- Encoding categorical variables such as `Sex` into numerical values for machine learning compatibility

These preprocessing steps helped clean the dataset and prepare it for Exploratory Data Analysis (EDA) and Linear Regression modeling.

###  Example



df['Age'] = df['Age'].fillna(df['Age'].median())

###  Lab 5 — Exploratory Data Analysis (EDA)

Exploratory Data Analysis (EDA) was performed to better understand the Titanic dataset and identify important patterns related to passenger survival.

###  Visualizations Created

- **Survival Count** → compares the number of survivors and non-survivors  

- **Passenger Class Distribution** → shows how passengers were distributed across classes  

- **Age Distribution** → analyzes passenger age ranges  

- **Fare Distribution** → visualizes ticket fare values and frequency  

- **Correlation Heatmap** → highlights relationships between numerical variables  

These visualizations helped identify trends and relationships between passenger characteristics and survival probability.

###  Lab 6 — Data Visualization with Streamlit

An interactive dashboard was developed using Streamlit to present the different stages of the project in a visual and user-friendly way.

###  Dashboard Features

- Dataset preview and structure visualization  
- Interactive filters for data exploration  
- Display of preprocessing results  
- Charts and graphical visualizations  
- Correlation heatmap between numerical variables  
- Machine Learning model results and predictions  

The dashboard made the project more interactive and allowed easier exploration of the Titanic dataset and the Linear Regression results.
###  Lab 7 — Statistical Modeling (Regression)

In this lab, Linear Regression was applied to analyze the relationship between passenger characteristics and survival probability on the Titanic dataset.

###  Features Used


['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
###  Target Variable
Survived
✔ Model Used
LinearRegression()
✔ Dataset Split

The dataset was divided into:

Training set
Testing set

The model was trained using the training data and evaluated using the testing data.

###  Evaluation Metrics

The model performance was evaluated using:

Mean Squared Error (MSE)
Mean Absolute Error (MAE)
R² Score
###  Example
mse = mean_squared_error(y_test, y_pred)

The regression model helped identify relationships between passenger information and survival outcomes.
