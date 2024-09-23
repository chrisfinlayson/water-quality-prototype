import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose


def load_data(sensor_file, weather_file):
    sensor_data = pd.read_csv(sensor_file, parse_dates=['Timestamp'])
    weather_data = pd.read_csv(weather_file, parse_dates=['Timestamp'])
    return sensor_data, weather_data

def clean_data(sensor_data, weather_data):
    # Remove outliers using Z-score
    for col in ['Flow_Rate', 'Nitrate_Level', 'Phosphate_Level']:
        sensor_data = sensor_data[np.abs(stats.zscore(sensor_data[col])) < 3]
    
    for col in ['Precipitation', 'Rainfall_Intensity', 'Temperature', 'Humidity', 'Wind_Speed']:
        weather_data = weather_data[np.abs(stats.zscore(weather_data[col])) < 3]
    
    # Interpolate missing values
    sensor_data = sensor_data.interpolate()
    weather_data = weather_data.interpolate()
    
    return sensor_data, weather_data

def merge_data(sensor_data, weather_data):
    return pd.merge(sensor_data, weather_data, on='Timestamp')

def calculate_statistics(data):
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    stats = data[numeric_columns].agg(['mean', 'median', 'std', 'min', 'max'])
    return stats

def plot_time_series(data, column, title):
    plt.figure(figsize=(12, 6))
    for sensor in data['Sensor_ID'].unique():
        sensor_data = data[data['Sensor_ID'] == sensor]
        plt.plot(sensor_data['Timestamp'], sensor_data[column], label=f'Sensor {sensor}')
    plt.title(title)
    plt.xlabel('Timestamp')
    plt.ylabel(column)
    plt.legend()
    plt.savefig(f'{column.lower()}_time_series.png')
    plt.close()

def plot_correlation_heatmap(data):
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    corr = data[numeric_columns].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Correlation Heatmap')
    plt.savefig('correlation_heatmap.png')
    plt.close()

def decompose_time_series(data, column, period):
    # Select data for a single sensor for decomposition
    sensor_data = data[data['Sensor_ID'] == data['Sensor_ID'].iloc[0]]
    
    # Perform seasonal decomposition
    decomposition = seasonal_decompose(sensor_data[column], model='additive', period=period)
    
    # Plot the decomposition
    plt.figure(figsize=(12, 10))
    plt.subplot(411)
    plt.plot(sensor_data['Timestamp'], decomposition.observed)
    plt.title(f'{column} - Observed')
    plt.subplot(412)
    plt.plot(sensor_data['Timestamp'], decomposition.trend)
    plt.title('Trend')
    plt.subplot(413)
    plt.plot(sensor_data['Timestamp'], decomposition.seasonal)
    plt.title('Seasonal')
    plt.subplot(414)
    plt.plot(sensor_data['Timestamp'], decomposition.resid)
    plt.title('Residual')
    plt.tight_layout()
    plt.savefig(f'{column.lower()}_decomposition.png')
    plt.close()

def create_lag_features(data, columns, lags):
    for col in columns:
        for lag in lags:
            data[f'{col}_lag_{lag}'] = data.groupby('Sensor_ID')[col].shift(lag)
    return data

def create_rolling_features(data, columns, windows):
    for col in columns:
        for window in windows:
            data[f'{col}_rolling_mean_{window}'] = data.groupby('Sensor_ID')[col].rolling(window=window).mean().reset_index(0, drop=True)
            data[f'{col}_rolling_std_{window}'] = data.groupby('Sensor_ID')[col].rolling(window=window).std().reset_index(0, drop=True)
    return data

def main():
    sensor_data, weather_data = load_data('sensor_data.csv', 'weather_data.csv')
    sensor_data, weather_data = clean_data(sensor_data, weather_data)
    merged_data = merge_data(sensor_data, weather_data)
    
    stats = calculate_statistics(merged_data)
    print("Basic Statistics:")
    print(stats)
    
    plot_time_series(merged_data, 'Nitrate_Level', 'Nitrate Levels Over Time')
    plot_time_series(merged_data, 'Phosphate_Level', 'Phosphate Levels Over Time')
    plot_time_series(merged_data, 'Flow_Rate', 'Flow Rate Over Time')
    
    plot_correlation_heatmap(merged_data)

    # Time series decomposition
    decompose_time_series(merged_data, 'Nitrate_Level', period=24*7)  # Weekly seasonality
    decompose_time_series(merged_data, 'Phosphate_Level', period=24*7)
    decompose_time_series(merged_data, 'Flow_Rate', period=24)  # Daily seasonality

    # Feature engineering
    columns_to_lag = ['Nitrate_Level', 'Phosphate_Level', 'Flow_Rate', 'Precipitation', 'Temperature']
    lags = [1, 3, 6, 12, 24]  # 1 hour, 3 hours, 6 hours, 12 hours, 1 day
    merged_data = create_lag_features(merged_data, columns_to_lag, lags)

    windows = [6, 12, 24, 48]  # 6 hours, 12 hours, 1 day, 2 days
    merged_data = create_rolling_features(merged_data, columns_to_lag, windows)

    # Save the processed data
    merged_data.to_csv('processed_data.csv', index=False)
    print("Processed data saved to 'processed_data.csv'")

    # Display information about new features
    print("\nNew features created:")
    print(merged_data.columns[merged_data.columns.str.contains('lag|rolling')])

if __name__ == "__main__":
    main()