import pandas as pd
df = pd.read_csv("diabetes.csv")
print(df.head())  # shows the first 5 rows of your dataset

print(df.info())        # shows column types and non-null counts
print(df.describe())    # shows mean, std, min, max for numeric columns
print(df['Outcome'].value_counts())  # shows how many 0s and 1s (non-diabetic vs diabetic)

X = df.drop('Outcome', axis=1)  # all columns except 'Outcome'
y = df['Outcome']               # target column

