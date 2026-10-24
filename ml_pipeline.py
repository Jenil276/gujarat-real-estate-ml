import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, accuracy_score
# pyrefly: ignore[missing-import]
import xgboost as xgb

def load_data():
    df = pd.read_csv("gujarat_real_estate.csv")
    X = df.drop(columns=["Price_INR", "Investment_Tier"])
    y_reg = df["Price_INR"]
    y_clf = df["Investment_Tier"]
    return df, X, y_reg, y_clf

def build_preprocessor():
    num_features = ["SqFt", "Bedrooms", "Property_Age_Years", "Distance_to_Transit_km", "Is_Gated_Society"]
    cat_features = ["City"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(drop="first"), cat_features)
        ])
    return preprocessor

def train_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "Ridge Regression": Pipeline(steps=[("prep", build_preprocessor()), ("model", Ridge(alpha=1.0))]),
        "Random Forest": Pipeline(steps=[("prep", build_preprocessor()), ("model", RandomForestRegressor(n_estimators=100, random_state=42))]),
        "XGBoost": Pipeline(steps=[("prep", build_preprocessor()), ("model", xgb.XGBRegressor(n_estimators=100, random_state=42))])
    }
    
    results = {}
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        results[name] = {"MAE (₹)": f"₹{mae:,.0f}", "Pipeline": pipeline}
        
    return results

def train_classification(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = Pipeline(steps=[("prep", build_preprocessor()), ("model", RandomForestClassifier(n_estimators=100, random_state=42))])
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    return clf, acc

def run_clustering(df):
    features = ["Price_INR", "SqFt"]
    X_clust = df[features]
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(StandardScaler().fit_transform(X_clust))
    return df