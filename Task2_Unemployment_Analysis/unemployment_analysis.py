
import pandas as pd
import matplotlib.pyplot as plt

# Load unemployment data
url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE"

df = pd.read_csv(url)

# Display first five rows
print("First five rows:")
print(df.head())

# Check dataset information
print("\nDataset information:")
df.info()

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Identify date and unemployment columns
date_col = df.columns[0]
rate_col = df.columns[1]

# Convert dates
df[date_col] = pd.to_datetime(df[date_col])

# Create unemployment trend graph
plt.figure(figsize=(12, 6))
plt.plot(df[date_col], df[rate_col])

plt.title("Unemployment Rate Over Time - USA")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.grid(True)
plt.tight_layout()

plt.savefig("unemployment_trend.png")
plt.show()

print("\nGraph saved successfully!")

# Summary statistics
print("\nUnemployment Summary:")
print(df[rate_col].describe())

# Find highest and lowest unemployment rates
highest = df.loc[df[rate_col].idxmax()]
lowest = df.loc[df[rate_col].idxmin()]

print("\nHighest Unemployment Rate:")
print(highest)

print("\nLowest Unemployment Rate:")
print(lowest)

# Histogram of unemployment rates
plt.figure(figsize=(10, 5))
plt.hist(df[rate_col].dropna(), bins=20, edgecolor="black")
plt.title("Distribution of Unemployment Rates - USA")
plt.xlabel("Unemployment Rate (%)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("unemployment_distribution.png")
plt.show()

print("\nBoth graphs have been saved!")