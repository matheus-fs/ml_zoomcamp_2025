#%%
import pandas as pd
import numpy as np

url = "https://raw.githubusercontent.com/alexeygrigorev/datasets/master/course_lead_scoring.csv"
df = pd.read_csv(url)

df.dtypes

#%%
categorical_columns = [
    "lead_source",
    "industry",
    "employment_status",
    "location"]

numerical_columns = [
    "number_of_courses_viewed",
    "annual_income",
    "interaction_count",
    "lead_score"
]
# %%
# Data preparation
df.isnull().sum()

df[categorical_columns] = df[categorical_columns].fillna("NA")
df[numerical_columns] = df[numerical_columns].fillna(0)
# %%
from sklearn.model_selection import train_test_split

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

print(len(df_train), len(df_val), len(df_test))

#%%