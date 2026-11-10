import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
# Load dataset
df = pd.read_csv("AirPassengers.csv")
# Convert Month to datetime
df["Month"] = pd.to_datetime(df["Month"])
# Set Month as index
df.set_index("Month", inplace=True)
# Display original data
print("Original Data:")
print(df.head())
# Plot original series
plt.figure(figsize=(12, 5))
plt.plot(df.index, df["Passengers"], marker="o")
plt.title("International Airline Passengers")
plt.xlabel("Year")
plt.ylabel("Passengers (Thousands)")
plt.grid(True)
plt.show()
# Apply log transformation
df["Log_Passengers"] = np.log(df["Passengers"])
print("\nLog Transformed Data:")
print(df.head())
# Plot log-transformed series
plt.figure(figsize=(12, 5))
plt.plot(df.index, df["Log_Passengers"], marker="o")
plt.title("Log-Transformed Airline Passenger Data")
plt.xlabel("Year")
plt.ylabel("Log(Passengers)")
plt.grid(True)

plt.show()
# Compare original and transformed variance
print("\nVariance Comparison:")
print("Original Variance:", df["Passengers"].var())
print("Log Transformed Variance:", df["Log_Passengers"].var())
# Compare original and log-transformed series
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
axes[0].plot(df.index, df["Passengers"])
axes[0].set_title("Original Airline Passenger Series")
axes[0].set_ylabel("Passengers")
axes[0].grid(True)
axes[1].plot(df.index, df["Log_Passengers"])
axes[1].set_title("Log-Transformed Airline Passenger Series")
axes[1].set_ylabel("Log(Passengers)")
axes[1].set_xlabel("Year")
axes[1].grid(True)
plt.tight_layout()
plt.show()
# Seasonal decomposition of log-transformed data
result = seasonal_decompose(
    df["Log_Passengers"],
    model="additive",
    period=12
)
result.plot()
plt.show()
