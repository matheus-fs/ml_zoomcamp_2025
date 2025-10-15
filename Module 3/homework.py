#%%
import pandas as pd
import numpy as np

url = "https://raw.githubusercontent.com/alexeygrigorev/datasets/master/course_lead_scoring.csv"
df = pd.read_csv(url)

df.dtypes
# %%
categorical_columns = ["lead_source", "industry", "employment_status", "location"]

numerical_columns = ["number_of_courses_viewed", "annual_income", "interaction_count", "lead_score"]

# Data preparation
df.isnull().sum()

df[categorical_columns] = df[categorical_columns].fillna("NA")
df[numerical_columns] = df[numerical_columns].fillna(0)
# %%
# Question 1: Mode for industry
print("Mode for the indutry column is ")
df["industry"].value_counts().head(1)
# %%
# Question 2:

df_numerical = df[numerical_columns].copy()
df_numerical.corr()

# %%
# Splitting the data
from sklearn.model_selection import train_test_split

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)

print(len(df_train), len(df_val), len(df_test))
# %%
# Question 3: Calculate the Mutual Information score between and other categorical variables
from sklearn.metrics import mutual_info_score

for i in categorical_columns:
    result = mutual_info_score(df_train["converted"], df_train[i])    
    print(f"Mutual score between converted and {i} - {round(result,2)}")

#%%
# Question 4: What accuracy did you get?

# Use one-hot encoding
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

dv = DictVectorizer(sparse=False)

train_dict = df_train[numerical_columns + categorical_columns].to_dict(orient='records')
X_train = dv.fit_transform(train_dict)
y_train = df_train.converted.values

val_dict = df_val[numerical_columns + categorical_columns].to_dict(orient='records')
X_val = dv.fit_transform(val_dict)
y_val = df_val.converted.values


model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict_proba(X_val)[:,1] #Get the second column which is the prediction os converting 
conversion = (y_pred >= 0.5).astype(int)

original_accuracy = (y_val == conversion).mean()

print(f"Base accuracy for the model: {round(original_accuracy,2)}")


# %%
# Question 5: 
all_parameters = numerical_columns + categorical_columns
results = []

for parameter in all_parameters:

    params = all_parameters.copy()
    params.remove(parameter)

    # Setup train values
    train_dict = df_train[params].to_dict(orient='records')
    X_train = dv.fit_transform(train_dict)
    y_train = df_train.converted.values

    # Setup validation values
    val_dict = df_val[params].to_dict(orient='records')
    X_val = dv.fit_transform(val_dict)
    y_val = df_val.converted.values

    # TRaining and fitting the model
    model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict_proba(X_val)[:,1] #Get the second column which is the prediction os converting 
    conversion = (y_pred >= 0.5).astype(int)

    accuracy = (y_val == conversion).mean()
    
    result = {
        "parameter": parameter,
        "accuracy": accuracy
    }

    results.append(result)

results = pd.DataFrame(results)
results["delta from original accuracy"] = abs(results["accuracy"] - original_accuracy)
results = results.sort_values("delta from original accuracy")
print(results)
# %%
# Question 6

# Setup train values
train_dict = df_train[all_parameters].to_dict(orient='records')
X_train = dv.fit_transform(train_dict)
y_train = df_train.converted.values

# Setup validation values
val_dict = df_val[all_parameters].to_dict(orient='records')
X_val = dv.fit_transform(val_dict)
y_val = df_val.converted.values


c_options = [0.01, 0.1, 1, 10, 100]

all_parameters = numerical_columns + categorical_columns
results = []

for c in c_options:

    # TRaining and fitting the model
    model = LogisticRegression(solver='liblinear', C=c, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict_proba(X_val)[:,1] #Get the second column which is the prediction os converting 
    conversion = (y_pred >= 0.5).astype(int)

    accuracy = (y_val == conversion).mean()
    
    result = {
        "c_option": c,
        "accuracy": round(accuracy,3)
    }

    results.append(result)

results = pd.DataFrame(results)
print(results)
# %%
