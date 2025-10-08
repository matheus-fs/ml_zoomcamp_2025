#%%
import pandas as pd
import numpy as np

# Question 1: Pandas version
print(f"Pandas Version - {pd.__version__}")

#%%
# Question 2: Record counts
url = "https://raw.githubusercontent.com/alexeygrigorev/datasets/master/car_fuel_efficiency.csv"
df = pd.read_csv(url)

df.shape[0]

# %%
# Question 3: How many fuel types are presented in the dataset
df["fuel_type"].value_counts().shape[0]


# %%
# Question 4: Missing values
null_count = df.isnull().sum()
null_count = null_count[null_count > 0]
print(len(null_count))

# %%
# Question 5: Max fuel efficiency
asian_cars = df[df["origin"] == "Asia"]

method1 = asian_cars["fuel_efficiency_mpg"].max()

asian_cars_sorted = asian_cars.sort_values(by = "fuel_efficiency_mpg", ascending=False).reset_index()
method2 = asian_cars_sorted["fuel_efficiency_mpg"][0]

assert method1 == method2
print(f"Max fuel efficiency - {method1}")
# %%
# Question 6:
null_count = df["horsepower"].isnull().sum()
most_frequent_value_count = df["horsepower"].value_counts().iloc[0]

mean_before = df.describe().loc["mean","horsepower"]
median_before = df["horsepower"].median()
most_frequent_value = df["horsepower"].mode()[0]
#%%
# Check median
position = int(len(df["horsepower"].dropna())/2)
#%%
median = df["horsepower"].dropna().sort_values().iloc[position]
print(f"{position} - {median}")

#%%
df["horsepower"] = df["horsepower"].fillna(most_frequent_value)
mean_after = df.describe().loc["mean","horsepower"]
median_after = df["horsepower"].median()

# Check median
position = int(len(df["horsepower"].dropna())/2)
median = df["horsepower"].dropna().sort_values().iloc[position]
print(f"{position} - {median}")




print(f"most_frequent_value - {most_frequent_value}")
print("----")
print(f"mean before - {mean_before}")
print(f"mean after - {mean_after}")
print(f"----")
print(f"median before - {median_before}")
print(f"median after - {median_after}")


most_frequent_value_count_after = df["horsepower"].value_counts().iloc[0]

assert null_count + most_frequent_value_count == most_frequent_value_count_after


#%%
# Question 7:
asian_cars = asian_cars[["vehicle_weight", "model_year"]]
asian_cars = asian_cars[:7]

X = asian_cars.values
XTX = X.T.dot(X)
inv_XTX = np.linalg.inv(XTX)

y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])

w = inv_XTX.dot(X.T)
w = w.dot(y)
print(sum(w))

# %%
