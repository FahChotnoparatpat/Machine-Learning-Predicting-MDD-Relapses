import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


df = pd.read_csv('preprocessed_data.csv')
# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Get the index of the 'Age' column
age_column_index = df.columns.get_loc('Age')

# Define the features (independent variables)
all_features = df.drop(df.columns[age_column_index], axis=1)
all_features = all_features.drop(['symptoms >2w', 'duration of episode', 'number of symptoms'], axis=1)

# Add a constant column to the features DataFrame
features = sm.add_constant(all_features)

# Create a DataFrame to store the VIF results
vif = pd.DataFrame()
vif["Feature"] = features.columns
vif["VIF"] = [variance_inflation_factor(features.values, i) for i in range(features.shape[1])]

# Print the VIF results
print(vif)