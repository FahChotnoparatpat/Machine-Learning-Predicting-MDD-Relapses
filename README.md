# Predicting the Chances of MDD (Major Depressive Disorder) Relapses through Machine Learning Models

This is an Applied Data Analysis project, where I created Logistic Regression, Decision Tree, and Random Forests model to predict whether a MDD patient will have a relapse.

In this directory, the files include

* **data.csv**: The columns include:
  * age (year)
  * gender (m/f)
  * preticipating events / acute events (y/n)
  * symptoms (categorical values)
  * duration of episode (days)
  * time to first relapse (days)
  * time to last relapse (days)
  * number of relapse (counts)
  * treatment (medicine / psychotherapy / none)

* **preprocessed_data.csv**: The columns include:
  * age (year)
  * gender (m/f)
  * preticipating events / acute events (y/n)
  * symptoms (dummie values)
  * treatment (medicine / psychotherapy / none)
  * relapse (y/n)

* **data_preprocessing**: This file takes the raw data (data.csv) as an input and produces preprocessed_data.csv

* **multicolinearity**: This file checks for multicolinearity of variables. The author has to manually exclude the variables in preprocessed_data.csv as necessary.


***Basline Approach:***

* **logistic_regression**: This file takes proprocessed data as an input, performs logistic regression and produces performance evaluation including significant predictors, test error, accuracy, sensitivity, specificity and AUC-ROC.

* **decision_tree**: This file takes proprocessed data as an input, performs decision tree algorithm and produces performance evaluation including accuracy, sensitivity, specificity and AUC-ROC. The author has to manually set the shrinkage values if changes are needed.

* **random_forests**: This file takes proprocessed data as an input, performs random forest algorithm and produces performance evaluation including sorted feature importance, accuracy, sensitivity, specificity and AUC-ROC.


***Combinatorial Approach:***

* **variable_combination**: This file determines the best feature combination for each model, based on the model's accuracy.

* **performance_evaluation**: This file performs logistic regression, decision tree, random forest algorithm with their best feature combination, and produces performance evaluation including accuracy, sensitivity, specificity and AUC-ROC.


You should access the file in this order.
1. data_preprocessing
2. multicolinearity
3. logistic_regression/ decision_tree/ random_forests
4. variable_combinations
5. performance_evaluation
