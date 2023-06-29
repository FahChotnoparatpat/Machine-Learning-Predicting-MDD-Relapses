import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Load the dataset
df = pd.read_csv('preprocessed_data.csv')

# Prepare the data
excluded_columns = ['symptoms >2w', 'duration of episode', 'number of symptoms']
df = df.drop(excluded_columns, axis=1)

x = df.drop(['relapse'], axis=1)
y = df['relapse'] 

# Split data into training set (90%) and test set (10%)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)

# Define a function for model fitting and prediction
def fit_and_predict_model(model, X_train, y_train, X_test):
    included_variables = [col for col in X_train.columns]
    combinations_list = list(combinations(included_variables, 5))
    best_combination = None
    best_accuracy = 0.0

    # Iterate over the combinations and evaluate the models
    for i, combination in enumerate(combinations_list):
        X_train_selected = X_train[list(combination)]
        X_test_selected = X_test[list(combination)]
        model.fit(X_train_selected, y_train)
        y_pred_selected = model.predict(X_test_selected)
        test_accuracy = accuracy_score(y_test, y_pred_selected)

        if test_accuracy > best_accuracy:
            best_combination = combination
            best_accuracy = test_accuracy

    return best_combination, best_accuracy

# Initialize the models
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}

# Iterate over the models
for model_name, model in models.items():
    best_combination, best_accuracy = fit_and_predict_model(model, X_train, y_train, X_test)

    # Print the best combination and its accuracy for the current model
    print(f"Best Combination for {model_name}: {best_combination}")
    print(f"Best Accuracy for {model_name}: {best_accuracy}")
    print()
