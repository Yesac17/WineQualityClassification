This project classifies red wine samples based on physicochemical attributes into quality categories using:

A Support Vector Classifier (SVC) with RBF kernel
A Random Forest Classifier

Both models are optimized using GridSearchCV and evaluated using confusion matrices, training/testing scores, and cross-validation metrics.


The dataset used is winequality-red.csv. It contains 1,599 observations of red wines, each with 11 features and a quality score from 0–10.

For classification purposes, the numeric quality score is converted into categorical labels:
≤ 5	low
6	medium
≥ 7	high

The predictors are standardized and split into 70% training / 30% testing sets.

Random Forest outperformed SVC both in terms of test accuracy (71.5% vs 64.4%) and cross-validation score.
Random Forest nearly perfectly memorizes the training set (99.3%), but generalizes decently to the test set.
SVC performs reasonably well but tends to underfit compared to the tree-based approach, especially for more separable classes.
Most misclassification occurs between neighboring classes, reflecting the natural overlap in wine quality ratings.
