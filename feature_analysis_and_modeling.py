import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import (SelectKBest, f_regression,
                                       mutual_info_regression)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler


def load_processed_data(file_path):
    return pd.read_csv(file_path, parse_dates=['Timestamp'])

def analyze_feature_importance(X, y, k=20):
    # Impute NaN values
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    y_imputed = imputer.fit_transform(y.values.reshape(-1, 1)).ravel()

    # Linear correlation
    f_scores, _ = f_regression(X_imputed, y_imputed)
    f_scores_normalized = f_scores / np.max(f_scores)
    
    # Mutual information
    mi_scores = mutual_info_regression(X_imputed, y_imputed)
    mi_scores_normalized = mi_scores / np.max(mi_scores)
    
    # Combine scores
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'F_Score': f_scores_normalized,
        'MI_Score': mi_scores_normalized
    })
    feature_importance['Avg_Score'] = (feature_importance['F_Score'] + feature_importance['MI_Score']) / 2
    feature_importance = feature_importance.sort_values('Avg_Score', ascending=False).reset_index(drop=True)
    
    # Select top k features
    selected_features = feature_importance.head(k)['Feature'].tolist()
    
    return feature_importance, selected_features

def plot_feature_importance(feature_importance):
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Avg_Score', y='Feature', data=feature_importance.head(20))
    plt.title('Top 20 Features by Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

def train_and_evaluate_model(X, y, model, model_name):
    # Impute NaN values
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    y_imputed = imputer.fit_transform(y.values.reshape(-1, 1)).ravel()

    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y_imputed, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    cv_mse = -cv_scores.mean()
    
    print(f"{model_name} Results:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R-squared Score: {r2:.4f}")
    print(f"Cross-validation MSE: {cv_mse:.4f}")
    print()

def main():
    data = load_processed_data('processed_data.csv')
    
    # Prepare features and target
    target = 'Nitrate_Level'
    features = [col for col in data.columns if col not in ['Timestamp', 'Sensor_ID', target]]
    X = data[features]
    y = data[target]
    
    # Analyze feature importance and select top features
    feature_importance, selected_features = analyze_feature_importance(X, y, k=20)
    plot_feature_importance(feature_importance)
    
    print("Top 10 Most Important Features:")
    print(feature_importance.head(10))
    print()
    
    # Use only selected features
    X_selected = X[selected_features]
    
    # Train and evaluate models
    linear_model = LinearRegression()
    rf_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    
    train_and_evaluate_model(X_selected, y, linear_model, "Linear Regression")
    train_and_evaluate_model(X_selected, y, rf_model, "Random Forest")

if __name__ == "__main__":
    main()