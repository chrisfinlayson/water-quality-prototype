from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def generate_sensor_data(start_date, end_date, num_sensors):
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')
    data = []
    
    for sensor_id in range(1, num_sensors + 1):
        for timestamp in date_range:
            flow_rate = np.sin(timestamp.hour / 24 * 2 * np.pi) * 5 + np.random.normal(10, 2)
            nitrate_level = np.random.normal(5, 1) + np.sin(timestamp.dayofyear / 365 * 2 * np.pi)
            phosphate_level = np.random.normal(2, 0.5) + np.sin(timestamp.dayofyear / 365 * 2 * np.pi) * 0.5
            
            data.append({
                'Timestamp': timestamp,
                'Sensor_ID': sensor_id,
                'Flow_Rate': max(0, flow_rate),
                'Nitrate_Level': max(0, nitrate_level),
                'Phosphate_Level': max(0, phosphate_level)
            })
    
    return pd.DataFrame(data)

def generate_weather_data(start_date, end_date):
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')
    data = []
    
    for timestamp in date_range:
        precipitation = max(0, np.random.exponential(1) if np.random.random() < 0.1 else 0)
        rainfall_intensity = precipitation / (1 if precipitation == 0 else np.random.uniform(0.5, 2))
        temperature = 15 + 10 * np.sin(timestamp.dayofyear / 365 * 2 * np.pi) + np.random.normal(0, 2)
        humidity = np.random.normal(70, 10)
        wind_speed = np.random.normal(10, 5)
        wind_direction = np.random.uniform(0, 360)
        
        data.append({
            'Timestamp': timestamp,
            'Precipitation': precipitation,
            'Rainfall_Intensity': rainfall_intensity,
            'Temperature': temperature,
            'Humidity': min(100, max(0, humidity)),
            'Wind_Speed': max(0, wind_speed),
            'Wind_Direction': wind_direction
        })
    
    return pd.DataFrame(data)

def main():
    # Reduce the time range to 1 month instead of a full year
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 1, 31)
    num_sensors = 2  # Reduce the number of sensors from 3 to 2

    print("Generating sensor data...")
    sensor_data = generate_sensor_data(start_date, end_date, num_sensors)
    print("Generating weather data...")
    weather_data = generate_weather_data(start_date, end_date)

    print("Saving data to CSV files...")
    sensor_data.to_csv('sensor_data.csv', index=False)
    weather_data.to_csv('weather_data.csv', index=False)
    print("Data generation complete.")

if __name__ == "__main__":
    main()