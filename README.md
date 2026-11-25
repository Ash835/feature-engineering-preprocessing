# Feature Engineering & Preprocessing Pipeline

A robust, reusable, and leak-free machine learning preprocessing pipeline built using Python and Scikit-Learn.

## Project Overview

This project demonstrates how to prepare complex tabular datasets for machine learning by implementing essential feature engineering and preprocessing techniques.

The **Adult Census Income Dataset** from OpenML is used to demonstrate numerical scaling, categorical encoding, missing value imputation, correlation analysis, and feature importance evaluation.

The project uses Scikit-Learn's `Pipeline` and `ColumnTransformer` to create an automated and reusable preprocessing workflow while preventing data leakage.

## Objectives

- Load and explore a real-world tabular dataset.
- Identify and handle missing values.
- Apply numerical feature scaling.
- Perform categorical feature encoding using One-Hot Encoding.
- Implement reusable preprocessing pipelines.
- Prevent data leakage through proper train-test splitting.
- Conduct correlation analysis.
- Evaluate feature importance using Mutual Information.

## Dataset

**Dataset:** Adult Census Income Dataset

**Source:** OpenML

**Target Variable:** `class`

The target variable represents two income categories:

- `<=50K`
- `>50K`

### Dataset Information

| Property | Value |
|---|---:|
| Total Records | 48,842 |
| Original Features | 14 |
| Numerical Features | 6 |
| Categorical Features | 8 |
| Target Classes | 2 |
| Training Records | 39,073 |
| Testing Records | 9,769 |

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- SciPy
- Jupyter Notebook

## Implementation Workflow

### 1. Data Loading and Exploration

- Loaded the Adult Census Income dataset using OpenML.
- Examined dataset structure, data types, and feature distributions.
- Identified missing values and duplicate records.

### 2. Train-Test Splitting

Split the dataset into:

- 80% Training Data
- 20% Testing Data

Used stratified sampling to maintain class distribution across both datasets.

### 3. Numerical Preprocessing

Implemented a Scikit-Learn Pipeline containing:

- `SimpleImputer(strategy="median")`
- `StandardScaler()`

### 4. Categorical Preprocessing

Implemented a separate Pipeline containing:

- `SimpleImputer(strategy="most_frequent")`
- `OneHotEncoder(handle_unknown="ignore")`

### 5. ColumnTransformer

Combined numerical and categorical preprocessing using Scikit-Learn's `ColumnTransformer`.

### 6. Data Leakage Prevention

- Split the dataset before preprocessing.
- Fitted preprocessing transformations exclusively on training data.
- Applied the learned transformations to testing data using `transform()`.

### 7. Correlation Analysis

- Calculated Pearson correlation coefficients.
- Visualized numerical feature relationships using a correlation heatmap.
- Examined highly correlated feature pairs.

### 8. Feature Importance

Used Mutual Information (`mutual_info_classif`) to identify informative features.

Continuous and discrete features were explicitly identified during Mutual Information calculation.

## Results

| Metric | Result |
|---|---:|
| Original Input Features | 14 |
| Processed Features | 105 |
| Continuous Features | 6 |
| Discrete Features | 99 |
| Training Records | 39,073 |
| Testing Records | 9,769 |
| Missing Values After Preprocessing | 0 |

### Top Informative Features

The Mutual Information analysis identified the following features among the top-ranked predictors:

- `fnlwgt`
- `marital-status_Married-civ-spouse`
- `relationship_Husband`
- `capital-gain`
- `age`
- `education-num`

Feature rankings are based on statistical dependence with the target and should not be interpreted as causal relationships.

## Project Structure

```text
Task-2-Feature-Engineering/
│
├── feature_engineering_preprocessing.ipynb
├── README.md
└── requirements.txt
```

## Installation and Execution

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
```

### 2. Navigate to the project directory

```bash
cd Task-2-Feature-Engineering
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open `feature_engineering_preprocessing.ipynb` and execute the notebook cells.

## Key Learnings

- Building reusable preprocessing pipelines.
- Handling missing values effectively.
- Encoding categorical variables.
- Scaling numerical features.
- Preventing data leakage in machine learning workflows.
- Analysing feature correlations.
- Evaluating feature importance using Mutual Information.

## Conclusion

This project successfully implements a reusable and leak-free preprocessing pipeline for a real-world tabular dataset.

The combination of Scikit-Learn Pipelines and ColumnTransformer provides an organised and efficient approach to preparing data for machine learning applications.