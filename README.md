**Telco Customer Churn Classifier Comparison**

A reproducible empirical study comparing classical and ensemble machine learning classification algorithms for customer churn prediction.

**Overview**

Customer churn prediction is a fundamental problem in customer relationship management and business analytics. The ability to identify customers who are likely to discontinue their service can help organizations develop targeted retention strategies and improve customer lifetime value.

This project presents a systematic comparison of multiple supervised classification algorithms applied to the Telco Customer Churn dataset. The study investigates how different algorithmic approaches perform under a consistent experimental framework and analyzes their relative strengths and limitations.

The analysis considers not only predictive performance but also model stability, computational efficiency, and sensitivity to data partitioning.

**Dataset**

The project uses the Telco Customer Churn dataset, which contains customer-level information related to demographics, subscribed services, account information, and churn status.

**Main Feature Categories**

* Customer demographics
* Account information
* Customer tenure
* Contract characteristics
* Internet and additional services
* Monthly and total charges
* Customer churn status

The target variable is:

```text
Churn
```

which indicates whether a customer discontinued the service.

**Experimental Methodology**

The project follows a structured machine learning workflow:

```text
Raw Dataset
     │
     ▼
Data Cleaning
     │
     ▼
Feature Preprocessing
     │
     ▼
Train / Validation / Test Split
     │
     ▼
Model Training
     │
     ▼
Validation and Model Evaluation
     │
     ▼
Performance Comparison
     │
     ▼
Statistical and Experimental Analysis
```

**Data Partitioning**

The dataset is divided into:

* Training set: 80%
* Validation set: 10%
* Test set: 10%

To investigate the sensitivity of model performance to data partitioning, experiments are conducted using multiple random seeds.

This allows the study to examine whether model performance remains consistent under different train-validation-test partitions.

**Classification Algorithms**

The following algorithms are evaluated under a unified experimental framework:

**Linear Models**

1. Logistic Regression
2. Linear Discriminant Analysis

**Distance-Based Models**

3. K-Nearest Neighbors

**Probabilistic Models**

4. Gaussian Naive Bayes

**Tree-Based Models**

5. Decision Tree

**Ensemble Models**

6. Random Forest
7. AdaBoost
8. Gradient Boosting

**Kernel-Based Models**

9. Support Vector Machine

**Neural Network Models**

10. Multi-Layer Perceptron

This selection provides a broad comparison across fundamentally different machine learning paradigms.

**Evaluation Framework**

Model performance is evaluated using multiple complementary metrics.

**Predictive Performance**

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

**Model Stability**

* Cross-validation mean
* Cross-validation standard deviation
* Performance variation across random seeds

**Computational Efficiency**

* Training time
* Prediction time

Using multiple metrics provides a more comprehensive assessment than relying on accuracy alone, particularly for churn prediction where identifying customers at risk of leaving may be more important than overall classification accuracy.

**Project Structure**

```text
telco-customer-churn-classifier-comparison/
│
├── data/
│   └── raw/
│       └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── notebooks/
│   ├── 01_logistic_regression.ipynb
│   ├── 02_lda.ipynb
│   ├── 03_knn.ipynb
│   ├── 04_gaussian_naive_bayes.ipynb
│   ├── 05_decision_tree.ipynb
│   ├── 06_random_forest.ipynb
│   ├── 07_svm.ipynb
│   ├── 08_adaboost.ipynb
│   ├── 09_gradient_boosting.ipynb
│   └── 10_mlp.ipynb
│
├── results/
│   ├── figures/
│   └── metrics/
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── splitting.py
│   ├── models.py
│   ├── evaluation.py
│   ├── visualization.py
│   └── experiment.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

**Reproducibility**

All experiments are conducted using explicitly defined random seeds to ensure reproducibility.

The project uses a modular source-code structure for data loading, preprocessing, dataset splitting, model construction, evaluation, visualization, and experiment management.

This separation improves:

* Reproducibility
* Maintainability
* Experimental consistency
* Code reusability

**Technologies**

* Python
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn
* Jupyter Notebook

**Project Status**

The project is currently under development. Final model comparisons, visualizations, and experimental findings will be added as the study progresses.

**Academic Context**

This project is being developed as part of an academic study in machine learning and applied mathematics, with a focus on empirical comparison of classification algorithms and reproducible computational experimentation.

**Research Objectives**

The main objectives of this study are to:

1. Implement and evaluate multiple classification algorithms using a consistent preprocessing and evaluation pipeline.
2. Compare model performance using several complementary evaluation metrics.
3. Investigate the effect of different random seeds and data partitions on predictive performance.
4. Analyze the strengths and limitations of linear, probabilistic, distance-based, tree-based, ensemble, kernel-based, and neural network models.
5. Identify suitable algorithms for customer churn classification based on predictive performance and computational considerations.
