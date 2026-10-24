import os
import pandas as pd
# pyrefly: ignore[missing-import]
import numpy as np

def generate_gujarat_data(n_samples=2500):
    np.random.seed(42)
    
    cities = ["Ahmedabad", "Surat", "Vadodara", "Rajkot", "Gandhinagar"]
    city_base_price = {"Ahmedabad": 4500, "Surat": 4200, "Vadodara": 3800, "Rajkot": 3500, "Gandhinagar": 5000} # Price per sqft
    
    # Generate realistic square footage bounded between 450 and 6000 sqft
    sqft = np.clip(np.random.normal(1500, 500, n_samples).astype(int), 450, 6000)
    transit_dist = np.clip(np.random.exponential(3, n_samples).round(1), 0.1, 25.0)

    data = {
        "City": np.random.choice(cities, n_samples),
        "SqFt": sqft,
        "Bedrooms": np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.05, 0.35, 0.40, 0.15, 0.05]),
        "Property_Age_Years": np.random.uniform(0, 30, n_samples).astype(int),
        "Distance_to_Transit_km": transit_dist,
        "Is_Gated_Society": np.random.choice([1, 0], n_samples, p=[0.7, 0.3])
    }
    
    df = pd.DataFrame(data)
    
    # Calculate price with added noise and feature modifiers
    df["Price_INR"] = (
        (df["SqFt"] * df["City"].map(city_base_price)) * 
        (1 + (df["Bedrooms"] * 0.05)) * 
        (1 - (df["Property_Age_Years"] * 0.01)) * # Depreciation
        (1 - (df["Distance_to_Transit_km"] * 0.02)) * # Distance penalty
        (1 + (df["Is_Gated_Society"] * 0.15)) # Premium for gated
    )
    
    # Add random market noise (fat tails)
    noise = np.random.normal(1, 0.1, n_samples)
    df["Price_INR"] = (df["Price_INR"] * noise).clip(lower=400000).round(-4).astype(int) # Round to nearest 10k
    
    # Classify investment tier (Classification Target)
    df["Investment_Tier"] = pd.qcut(df["Price_INR"], q=3, labels=["Affordable", "Mid-Range", "Luxury"], duplicates="drop")
    
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gujarat_real_estate.csv")
    df.to_csv(output_path, index=False)
    print(f"Generated {output_path} with {n_samples} records.")

if __name__ == "__main__":
    generate_gujarat_data()