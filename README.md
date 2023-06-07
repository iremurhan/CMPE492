# Enhancing Cancer Diagnosis through Machine Learning and Genomic Data Analysis

This Python code processes and evaluates datasets for different cancer types. It calculates the feature importances according to different classifiers, using techniques such as LIME for feature explanation and Principle Component Analysis (PCA) for feature selection.

## How to Use

### Install the necessary Python libraries:

```shell
pip install pandas numpy sklearn lime
```

### Structure your directory as follows:

```
.
├── main.py
└── datasets
    ├── liver_cancer
    │   ├── dataset1.csv
    │   ├── dataset2.csv
    │   └── ...
    └── breast_cancer
        ├── dataset1.csv
        ├── dataset2.csv
        └── ...
```

- `main.py`: This is where the main Python code resides.
- `datasets`: This is the directory where all your datasets are stored. This directory should include one subdirectory for each cancer type (e.g., `liver_cancer`, `breast_cancer`).
- `liver_cancer` and `breast_cancer` directories: These directories should contain the .csv files of the datasets related to each cancer type.

### Run the Python Code:

Run the Python script `main.py` to start the process.

This will:

- Load and process each cancer dataset.
- Perform feature explanation with LIME for Logistic Regression and Random Forest classifiers.
- Perform feature selection with PCA for Logistic Regression and Random Forest classifiers.
- Output the most important features for each classifier in each cancer type dataset.

## Notes:

- Adjust the `num_features_to_select` and `n_components` (for PCA in the LIME function) based on your preference.
- The functions provided in this code are basic and might need adjustments based on the specific requirements of your datasets.
- Running time of this code changes based on the performance of your computer, data and feature sizes. For liver cancer databases 'Liver_GSE14520_U133A.csv' and 'Liver_GSE62232.csv' it takes around 50 minutes.