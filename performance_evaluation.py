import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv('preprocessed_data.csv')
df.columns = df.columns.str.strip()

random_forest_columns = ['poor concentration', 'gender', 'treatment', 'difficulty sleeping', 'age', 'relapse']

# Create a new DataFrame with only the included columns
random_forest_df = df[random_forest_columns]

random_forest_x = random_forest_df.drop(['relapse'], axis=1)
random_forest_y = random_forest_df['relapse']

# Split data into training set (90%) and test set (10%)
random_forest_x_train, random_forest_x_test, random_forest_y_train, random_forest_y_test = train_test_split(random_forest_x, random_forest_y, test_size=0.1, random_state=1)

# Random Forests
rf_classifier = RandomForestClassifier(n_estimators=100)
rf_classifier.fit(random_forest_x_train, random_forest_y_train)
rf_predictions = rf_classifier.predict(random_forest_x_test)
rf_accuracy = accuracy_score(random_forest_y_test, rf_predictions)
rf_cm = confusion_matrix(random_forest_y_test, rf_predictions)
rf_specificity = rf_cm[0, 0] / (rf_cm[0, 0] + rf_cm[0, 1])
rf_sensitivity = rf_cm[1, 1] / (rf_cm[1, 0] + rf_cm[1, 1])
rf_auc_roc = roc_auc_score(random_forest_y_test, rf_predictions)
print("Random Forests - Accuracy:", rf_accuracy)
print("Random Forests - Sensitivity:", rf_sensitivity)
print("Random Forests - Specificity:", rf_specificity)
print("Random Forests - AUC-ROC:", rf_auc_roc)

decision_tree_columns = ['age', 'precipitating factors', 'decreased activities', 'hopelessness',  'difficulty sleeping', 'relapse']

# Create a new DataFrame with only the included columns
decision_tree_df = df[decision_tree_columns]

decision_tree_x = decision_tree_df.drop(['relapse'], axis=1)
decision_tree_y = decision_tree_df['relapse']

# Split data into training set (90%) and test set (10%)
decision_tree_x_train, decision_tree_x_test, decision_tree_y_train, decision_tree_y_test = train_test_split(decision_tree_x, decision_tree_y, test_size=0.1, random_state=1)

# Decision Tree
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(decision_tree_x_train, decision_tree_y_train)
dt_predictions = dt_classifier.predict(decision_tree_x_test)
dt_accuracy = accuracy_score(decision_tree_y_test, dt_predictions)
dt_cm = confusion_matrix(decision_tree_y_test, dt_predictions)
dt_specificity = dt_cm[0, 0] / (dt_cm[0, 0] + dt_cm[0, 1])
dt_sensitivity = dt_cm[1, 1] / (dt_cm[1, 0] + dt_cm[1, 1])
dt_auc_roc = roc_auc_score(decision_tree_y_test, dt_predictions)
print("Decision Tree - Accuracy:", dt_accuracy)
print("Decision Tree - Sensitivity:", dt_sensitivity)
print("Decision Tree - Specificity:", dt_specificity)
print("Decision Tree - AUC-ROC:", dt_auc_roc)

logreg_columns = ['age', 'gender', 'family history', 'precipitating factors', 'co-morbidities', 'relapse']

# Create a new DataFrame with only the included columns
logreg_df = df[logreg_columns]

logreg_x = logreg_df.drop(['relapse'], axis=1)
logreg_y = logreg_df['relapse']

# Split data into training set (90%) and test set (10%)
logreg_x_train, logreg_x_test, logreg_y_train, logreg_y_test = train_test_split(logreg_x, logreg_y, test_size=0.1, random_state=1)

# Logistic Regression
logreg_classifier = LogisticRegression(max_iter=1000)
logreg_classifier.fit(logreg_x_train, logreg_y_train)
logreg_predictions = logreg_classifier.predict(logreg_x_test)
logreg_accuracy = accuracy_score(logreg_y_test, logreg_predictions)
logreg_cm = confusion_matrix(logreg_y_test, logreg_predictions)
logreg_specificity = logreg_cm[0, 0] / (logreg_cm[0, 0] + logreg_cm[0, 1])
logreg_sensitivity = logreg_cm[1, 1] / (logreg_cm[1, 0] + logreg_cm[1, 1])
logreg_auc_roc = roc_auc_score(logreg_y_test, logreg_predictions)
print("Logistic Regression - Accuracy:", logreg_accuracy)
print("Logistic Regression - Sensitivity:", logreg_sensitivity)
print("Logistic Regression - Specificity:", logreg_specificity)
print("Logistic Regression - AUC-ROC:", logreg_auc_roc)
