#%%
import pandas as pd
import numpy as np

url = "https://raw.githubusercontent.com/alexeygrigorev/datasets/master/car_fuel_efficiency.csv"
df = pd.read_csv(url)

df.shape[0]

df = df[[
    "engine_displacement",
    "horsepower",
    "vehicle_weight",
    "model_year",
    "fuel_efficiency_mpg"]]

#%%
# Question 1
df.isnull().sum()
# %%
# Question 2
df["horsepower"].median()
# %%
n = len(df)

n_val = int(n*0.2)
n_test = int(n*0.2)
n_train = n - n_val - n_test

assert n == (n_val + n_train + n_test)

idx = np.arange(n)

np.random.seed(42)
np.random.shuffle(idx)

df_train = df.iloc[idx[n_train:]]
df_val = df.iloc[idx[n_train:n_train+n_val]]
df_test = df.iloc[idx[n_train+n_val:]]

df_train_zeros = df_train.copy()
df_train_zeros["horsepower"] = df_train_zeros["horsepower"].fillna(0)

df_train_mean = df_train.copy()
mean = df_train_mean["horsepower"].mean()
df_train_mean["horsepower"] = df_train_mean["horsepower"].fillna(mean)

print(f"Horsepower mean - {mean}")
print(f"Horsepower mean substuting na with zeros - {df_train_zeros["horsepower"].mean()}")
print(f"Horsepower mean substuting na with mean - {df_train_mean["horsepower"].mean()}")

# %%
def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)
    
    return w[0], w[1:]

def rmse(y, y_pred):
    error = y_pred - y
    mse = (error ** 2).mean()
    return np.sqrt(mse)


#%%
# Training with the dataframe filled with the mean
columns = [
    "engine_displacement",
    "horsepower",
    "vehicle_weight",
    "model_year"
]

X_train_mean = df_train_mean[columns].to_numpy()
X_train_zeros = df_train_zeros[columns].to_numpy()
X_val_mean = df_val[columns].fillna(mean).to_numpy()
X_val_zeros = df_val[columns].fillna(0).to_numpy()


y_train = df_train["fuel_efficiency_mpg"].to_numpy()
y_val = df_val["fuel_efficiency_mpg"].to_numpy()

# Training using mean
w_0, w = train_linear_regression(X_train_mean, y_train)
y_pred = w_0 + X_val_mean.dot(w)

score_mean = rmse(y_val, y_pred)
score_mean = round(score_mean,2)
print(f"score using mean when replacing na - {score_mean}")

#Train using zeros
w_0, w = train_linear_regression(X_train_zeros, y_train)
y_pred = w_0 + X_val_zeros.dot(w)

score_zeros = rmse(y_val, y_pred)
score_zeros = round(score_zeros,2)
print(f"score using zeros when replacing na - {score_zeros}")


#%%
#Question 4

r = [0, 0.01, 0.1, 1, 5, 10, 100]

def train_linear_regression_reg(X, y, r=0.001):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX = XTX + r * np.eye(XTX.shape[0])

    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)
    
    return w[0], w[1:]

for i in r:
    #Train using zeros
    w_0, w = train_linear_regression_reg(X_train_zeros, y_train, i)
    
    
    y_pred = w_0 + X_val_zeros.dot(w)

    score = rmse(y_val, y_pred)
    # score = round(score,2)

    print(f"Regularization used: {i} - result: {score}")

    
    
    
    




# %%
