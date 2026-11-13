# Experiment 03 – Log Transformation of International Airline Passenger Data

### Name: NITHYA T
### Register No: 2305001023

## Aim
To apply a logarithmic transformation to the International Airline Passenger time series and
compare the original and transformed data.

## Algorithm
1. Start the program.
2. Import Pandas, NumPy, Matplotlib, and seasonal decomposition functions.
3. Load the AirPassengers.csv dataset.
4. Convert the Month column to datetime and set it as the index.
5. Display and plot the original passenger series.
6. Apply logarithmic transformation using np.log().
7. Display and plot the log-transformed series.
8. Calculate and compare the variance before and after transformation.
9. Plot the original and log-transformed series for comparison.
10. Perform seasonal decomposition on the log-transformed series with period 12.
11. Observe the change in variance and seasonal behavior.
12. Stop the program.


## Program
File: `experiment_03.py`

```python
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
```

## Output
<img width="975" height="445" alt="image" src="https://github.com/user-attachments/assets/b00ab37c-0cd8-4fcd-9f3c-9c99105a6d67" />

<img width="985" height="460" alt="image" src="https://github.com/user-attachments/assets/31f08d8c-2a76-4a85-816e-86bee8426c76" />

<img width="980" height="695" alt="image" src="https://github.com/user-attachments/assets/fa69b07b-be43-4f3e-992f-95a5d728eb3e" />


## Observation
The original passenger series shows increasing variation as the passenger count rises. After applying
the logarithmic transformation, the scale is compressed and the variance is greatly reduced. The
transformed series has more stable variability, while the seasonal pattern remains clearly visible.

## Result
Thus, the logarithmic transformation was successfully applied to the International Airline Passenger
time series. The original and transformed series were compared, variance was reduced, and seasonal
decomposition was performed successfully.
