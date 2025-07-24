import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("train.csv")

print("Original Data Shape:", df.shape)
print(df[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'SalePrice']].head())

df = df[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'SalePrice']].dropna()

X = df[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
y = df['SalePrice']

sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📈 Model Evaluation:")
print("Mean Squared Error:", mse)
print("R² Score:", r2)

print("\n📊 Feature Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef}")

plt.scatter(y_test, y_pred, alpha=0.6, color='green')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.show()

new_house = pd.DataFrame([[2000, 3, 2]], columns=['GrLivArea', 'BedroomAbvGr', 'FullBath'])
predicted_price = model.predict(new_house)
print("\n💡 Predicted price for 2000 sqft, 3 beds, 2 baths: ₹", round(predicted_price[0]))
