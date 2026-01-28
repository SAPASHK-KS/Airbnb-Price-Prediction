import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("airbnb(1).csv")

# Keep only required columns
df = df[['bathrooms', 'beds', 'guests', 'reviews', 'rating', 'bedrooms', 'price']]

# Remove commas and convert to numeric
for col in ['bathrooms', 'beds', 'guests', 'reviews', 'rating', 'bedrooms', 'price']:
    df[col] = df[col].astype(str).str.replace(',', '', regex=True)
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill missing values
df.fillna(df.median(numeric_only=True), inplace=True)

# Features & target
X = df[['bathrooms', 'beds', 'guests', 'reviews', 'rating', 'bedrooms']]
y = df['price']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("✅ Model trained successfully with clean numeric data!")
