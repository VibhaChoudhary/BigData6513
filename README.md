## Table of contents
* [General info](#general-info)
* [Dataset](#dataset)
* [Process](#process)
* [Results](#results)
* [Definitions](#definitions)


## General info
The goal is to predict employee attrition by running ML classifiers in Spark (Java) and R.

## Dataset
The data set presents an employee survey from IBM, indicating if there is attrition or not. The dataset contains 35 variables along with Attrition variable. (Data source: https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset.)

## Process
### Step 1: Predict employee attrition by running any three ML classifiers in Spark (Java) and R (“mlr”).
Classifiers that are run are:
*	Logistic Regression Classification 
*	Random Forest Classification
*	Gradient Boost Classification
### Step 2: Run 5 classifiers parallelly on the same dataset in both Java and R and compare the results.
*	Logistic Regression Classification 
*	Random Forest Classification
*	Gradient Boost Classification
*	Decision Tree Classification
*	Naïve Bayes Classification
<br />
Parallel execution in R is done using doParallel package.
<br />
Parallel execution in Java is done using Java 8 ParallelStream and foreach.

## Results
![alt text](https://raw.githubusercontent.com/VibhaChoudhary/EmployeeAttritionPredictor/master/ClassifierResults.png)

## Definitions
*	TP: True Positive
<br />Number of employees correctly identified as left (Actual = yes, Predicted = yes).
*	TN: True Negative
<br />Number of employees correctly identified as not left (Actual = no, Predicted = no).
* FP: False positive
<br />Number of employees incorrectly identified as left (Actual = yes, Predicted = no).
*	FN: False Negative
<br />Number of employees incorrectly identified as not left (Actual = no, Predicted = yes).
*	Accuracy: reflects the classifier’s overall prediction correctness.
<br />ACC = (TP + TN) / (TP + TN + FP + FN)
*	Test Error: reflects the classifier’s overall prediction error.
<br />Test Error = 1 – ACC
* AUC:  stands for "Area under the ROC Curve." AUC is the probability that the model ranks a random positive example more highly than a random negative example. ROC curve plots two parameters:
<br />True Positive Rate = TP / (TP + FN)
<br />False Positive Rate = FP / (FP + TN)
