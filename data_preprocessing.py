import csv
import pandas as pd

symptoms = []
# open the CSV file
with open('data.csv', 'r') as csvfile:
    # create a CSV reader object
    reader = csv.reader(csvfile)
    # loop through each row in the CSV file
    for row in reader:
        symptoms.append(row[5])

df = pd.read_csv('data.csv')

# Insert the new columns
df.insert(loc=6, column='loss of interest', value=None)
df.insert(loc=7, column='decreased activities', value=None)
df.insert(loc=8, column='poor concentration', value=None)
df.insert(loc=9, column='excessive guilt', value=None)
df.insert(loc=10, column='hopelessness', value=None)
df.insert(loc=11, column='thoughts of suicide', value=None)
df.insert(loc=12, column='difficulty sleeping', value=None)
df.insert(loc=13, column='loss of appetite', value=None)
df.insert(loc=14, column='fatigue', value=None)
df.insert(loc=15, column='number of symptoms', value=None)
df.insert(loc=16, column='severity', value=None)

start_index = 2
# Create dummy columns for each symptom
for index, row_item in enumerate(symptoms[start_index+1:], start=start_index):
    symptom = row_item[1:].split()
    symptom = str(symptom)

    loss_of_interest = 1 if '1' in symptom else 0  
    decreased_activities = 1 if '2' in symptom else 0
    poor_con = 1 if '3' in symptom else 0
    excessive_guilt = 1 if '4' in symptom else 0
    hopelessness = 1 if '5' in symptom else 0
    thoughts_of_suicide = 1 if '6' in symptom else 0
    difficulty_sleeping = 1 if '7' in symptom else 0
    loss_of_appetite = 1 if '8' in symptom else 0
    fatigue = 1 if '9' in symptom else 0
    
    # Update the corresponding row values in the new columns
    df.at[index, 'loss of interest'] = loss_of_interest
    df.at[index, 'decreased activities'] = decreased_activities
    df.at[index, 'poor con'] = poor_con
    df.at[index, 'excessive guilt'] = excessive_guilt
    df.at[index, 'hopelessness'] = hopelessness
    df.at[index, 'thoughts of suicide'] = thoughts_of_suicide
    df.at[index, 'difficulty sleeping'] = difficulty_sleeping
    df.at[index, 'loss of appetite'] = loss_of_appetite
    df.at[index, 'fatigue'] = fatigue

# Calculate the number of symptoms for each patient
row_sums = df.iloc[start_index:, 6:15].sum(axis=1)
df['number of symptoms'] = row_sums


# Calculate severity
for index, row_item in enumerate(df['number of symptoms'][start_index:], start=start_index):
    num_symptoms = row_item
    if num_symptoms <= 3:
        df.at[index, 'severity'] = '1'
    elif num_symptoms <= 6:
        df.at[index, 'severity'] = '2'
    elif num_symptoms <= 9:
        df.at[index, 'severity'] = '3'


# Drop rows that do not meet requirements

# Store the row data before dropping
rows_stored = df.loc[1].copy()      # This is the descriptions of each column. I won't add them back but I want to store them just in case
df = df[2:].reset_index(drop=True)

df['time to first relapse'] = df['time to first relapse'].astype(int)
df['number of relapses'] = df['number of relapses'].astype(int)

# Drop records where 'time to first relapse' < 60 and number of relapses != 0
mask = (df['time to first relapse'] < 60) & (df['number of relapses'] != 0)
df = df.reset_index(drop=True)

# Reformat the relapses data
# Create the 'relapse' column and set initial values to 0
df['relapse'] = 0

# Set 'relapse' to 1 where 'number of relapses' is greater than 0
df.loc[df['number of relapses'] > 0, 'relapse'] = 1

columns_to_drop = ['time to first relapse', 'time to last relapse', 'number of relapses']
df = df.drop(columns=columns_to_drop)

# Save the updated DataFrame to a new CSV file
df.to_csv('preprocessed_data.csv', index=False)
