# Lab 3: Credit Card Fraud Detection and Insurance Cost Prediction

**Name:** Sukhchain Singh  
**Student Number:** 9111541  

## Overview

This lab covers two supervised learning tasks:

- **Part A:** Use logistic regression to classify credit card transactions as normal or fraudulent.
- **Part B:** Use linear regression to predict insurance charges.

Both parts include data quality checks, preprocessing, training data analysis, a 70/30 train-test split, model evaluation, and explanations of the findings.

## Datasets

Download and extract the following datasets into the `data` folder:

| Dataset | File | Source |
| --- | --- | --- |
| Credit Card Fraud Detection | `creditcard.csv` | [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| Insurance Costs | `insurance.csv` | [Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance) |

The notebook expects both CSV files inside a `data` folder located beside the notebook.

## Part A: Credit Card Fraud Detection

### Fraud Detection Approach

- Checked data types, missing values, duplicates, and class labels.
- Removed 1,081 exact duplicate copies, leaving 283,726 transactions.
- Used a stratified 70/30 split to preserve the fraud proportion.
- Explored the training data using summary statistics and a class distribution graph.
- Standardized features using values learned from the training data only.
- Trained logistic regression and compared it with an always-normal baseline.

### Fraud Detection Test Results

| Metric | Result |
| --- | ---: |
| Accuracy | 99.91% |
| Fraud precision | 84.04% |
| Fraud recall | 55.63% |
| Fraud F1-score | 0.6695 |
| Fraudulent transactions detected | 79 |
| Fraudulent transactions missed | 63 |
| False alarms | 15 |

Although accuracy was high, the model missed many fraudulent transactions. This shows why accuracy alone is not enough for a highly imbalanced dataset.

## Part B: Insurance Cost Prediction

### Insurance Cost Prediction Approach

- Checked missing values, duplicates, category labels, and numerical ranges.
- Removed one exact duplicate copy, leaving 1,337 records.
- Used a 70/30 train-test split.
- Explored charges, age, and smoking status in the training data.
- One-hot encoded categorical features using training data only.
- Trained linear regression and compared it with a training-mean baseline.
- Examined actual versus predicted charges and residuals.

### Test Results

| Metric | Result |
| --- | ---: |
| MAE | 4,181.82 |
| RMSE | 6,240.21 |
| R² | 0.7724 |
| Negative predictions | 7 |

The model performed better than the baseline but underestimated several high charges and produced seven unrealistic negative predictions. R² is not a percentage accuracy score.

## Tools Used

- Python
- Jupyter Notebook and VS Code
- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

## How to Run

1. Download or clone this repository.
2. Download the two datasets from the links above.
3. Extract `creditcard.csv` and `insurance.csv` into the `data` folder.
4. Open the notebook in VS Code or Jupyter.
5. Select a Python environment with the required libraries installed.
6. Restart the kernel and run all cells from top to bottom.

To create a Conda environment with the required packages:

~~~bash
conda create -n lab3 python=3.11 pandas numpy matplotlib seaborn scikit-learn ipykernel nbconvert -y
conda activate lab3
~~~

## Limitations

- Results are based on one train-test split.
- Duplicate removal assumes that retaining one copy of identical rows is appropriate. Unique identifiers were unavailable to verify this.
- The fraud model misses a substantial proportion of fraudulent transactions.
- The insurance model does not fully capture the patterns in charges and can predict negative values.
- These models are educational examples, not production-ready systems.

## Submission Files

The final notebook and HTML export must follow this naming format:

- `Sukhchain Singh_9111541_Lab3.ipynb`
- `Sukhchain Singh_9111541_Lab3.html`

The notebook must include all executed cell outputs. Both files must be submitted on eConestoga.

## References

- [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [Insurance dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- [GitHub Pages Quickstart](https://docs.github.com/en/pages/quickstart)
