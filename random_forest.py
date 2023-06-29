import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score,  confusion_matrix

# Load the dataset
df = pd.read_csv('preprocessed_data.csv')

# create imputer to replace NaN with median
imputer = SimpleImputer(strategy='median')

# Prepare the data
excluded_columns = ['symptoms >2w', 'duration of episode', 'number of symptoms']
df = df.drop(excluded_columns, axis=1)

x = df.drop(['relapse'], axis=1)
y = df['relapse']

# Split data into training set (90%) and test set (10%)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)

# Fit imputer on training data and transform training and test data
X_train_imputed = imputer.fit_transform(x_train)
X_test_imputed = imputer.transform(x_test)

# Initialize and fit the random forest classifier
rf_classifier = RandomForestClassifier(random_state=1)
rf_classifier.fit(X_train_imputed, y_train)

# Predict on test data
y_pred = rf_classifier.predict(X_test_imputed)

# Evaluate the model
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = rf_classifier.feature_importances_
feature_names = x_train.columns

# Sort feature importance in descending order
indices = np.argsort(feature_importance)[::-1]
sorted_feature_importance = feature_importance[indices]
sorted_feature_names = feature_names[indices]

# Print the feature importance
print("Feature Importance:")
for i in range(len(sorted_feature_names)):
    print(f"{sorted_feature_names[i]}: {sorted_feature_importance[i]}")

rf_classifier = RandomForestClassifier(n_estimators=100)
rf_classifier.fit(x_train, y_train)
rf_predictions = rf_classifier.predict(x_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_cm = confusion_matrix(y_test, rf_predictions)
rf_specificity = rf_cm[0, 0] / (rf_cm[0, 0] + rf_cm[0, 1])
rf_sensitivity = rf_cm[1, 1] / (rf_cm[1, 0] + rf_cm[1, 1])
rf_auc_roc = roc_auc_score(y_test, rf_predictions)
print("Random Forests - Accuracy:", rf_accuracy)
print("Random Forests - Sensitivity:", rf_sensitivity)
print("Random Forests - Specificity:", rf_specificity)
print("Random Forests - AUC-ROC:", rf_auc_roc)