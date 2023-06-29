import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix


# Load the dataset
df = pd.read_csv('preprocessed_data.csv')

# create imputer to replace NaN with median
imputer = SimpleImputer(strategy='median')

# Prepare the data
excluded_columns = ['symptoms >2w', 'duration of episode', 'number of symptoms']
df = df.drop(excluded_columns, axis=1)

x = df.drop(['relapse'], axis=1)
y = df['relapse'] 


# split data into training set (90%) and test set (10%)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)

# fit imputer on training data and transform training and test data
X_train_imputed = imputer.fit_transform(x_train)
X_test_imputed = imputer.transform(x_test)

# fit logistic regression model on training data with imputed values
X_train_const = sm.add_constant(X_train_imputed)
logreg = sm.Logit(y_train, X_train_const).fit()

# print out the summary of the model
print(logreg.summary())

# get the p-values for each coefficient
coef_pvalues = logreg.pvalues[1:]

# get the indices of the significant coefficients
significant_indices = np.where(coef_pvalues < 0.05)[0]


# get the names of the significant predictors
significant_predictors = x_train.columns[significant_indices]

# refit logistic regression model on significant predictors
X_train_selected = x_train[significant_predictors]
X_test_selected = x_test[significant_predictors]

logreg_selected = LogisticRegression()
logreg_selected.fit(X_train_selected, y_train)

# predict mpg01 on test data with significant predictors
y_pred_selected = logreg_selected.predict(X_test_selected)
print(f'Significant Predictors: {significant_predictors}')

test_error = 1 - accuracy_score(y_test, y_pred_selected)
print(f'Test Error: {test_error}')

logreg_classifier = LogisticRegression(max_iter=1000)
logreg_classifier.fit(x_train, y_train)
logreg_predictions = logreg_classifier.predict(x_test)
logreg_accuracy = accuracy_score(y_test, logreg_predictions)
logreg_cm = confusion_matrix(y_test, logreg_predictions)
logreg_specificity = logreg_cm[0, 0] / (logreg_cm[0, 0] + logreg_cm[0, 1])
logreg_sensitivity = logreg_cm[1, 1] / (logreg_cm[1, 0] + logreg_cm[1, 1])
logreg_auc_roc = roc_auc_score(y_test, logreg_predictions)
print("Logistic Regression - Accuracy:", logreg_accuracy)
print("Logistic Regression - Sensitivity:", logreg_sensitivity)
print("Logistic Regression - Specificity:", logreg_specificity)
print("Logistic Regression - AUC-ROC:", logreg_auc_roc)