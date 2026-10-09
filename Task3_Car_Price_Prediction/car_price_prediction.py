
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Create a sample car dataset
data = {
    "Year": [2018, 2019, 2020, 2017, 2021, 2016, 2022, 2019, 2020, 2018,
             2017, 2021, 2022, 2016, 2020, 2019, 2018, 2021, 2017, 2022],
    "Kilometers_Driven": [30000, 25000, 20000, 50000, 10000, 60000, 5000,
                          35000, 18000, 40000, 55000, 12000, 8000, 70000,
                          22000, 28000, 45000, 15000, 65000, 6000],
    "Engine_CC": [1200, 1500, 1200, 1000, 1600, 1000, 1800, 1500, 1200, 1400,
                  1000, 1600, 1800, 1000, 1200, 1500, 1400, 1600, 1000, 1800],
    "Price": [450000, 600000, 650000, 300000, 900000, 250000, 1200000,
              550000, 700000, 400000, 280000, 950000, 1300000, 200000,
              680000, 580000, 420000, 980000, 230000, 1250000]
}

df = pd.DataFrame(data)

print("First five rows:")
print(df.head())

print("\nDataset information:")
print(df.info())

# Features and target
X = df[["Year", "Kilometers_Driven", "Engine_CC"]]
y = df["Price"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict car prices
predictions = model.predict(X_test)

print("\nActual prices:", list(y_test))
print("Predicted prices:", predictions.round(0).tolist())

print("\nMean Absolute Error:", round(mean_absolute_error(y_test, predictions), 2))
print("R2 Score:", round(r2_score(y_test, predictions), 3))

# Plot actual vs predicted prices
plt.figure(figsize=(8, 5))
plt.scatter(y_test, predictions)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")
plt.tight_layout()
plt.savefig("car_price_prediction.png")
plt.show()

print("\nGraph saved as car_price_prediction.png")