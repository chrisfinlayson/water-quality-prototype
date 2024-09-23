import json
import logging
import os
from datetime import datetime

import dash
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from dash import dcc, html
from dash.dependencies import Input, Output, State
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
LOW_THRESHOLD = 5
HIGH_THRESHOLD = 10
CACHE_DIR = 'cache'

# Ensure cache directory exists
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# Create Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1("Nitrate Pollution Analysis Dashboard"),
    html.Div([
        html.Button('Load and Process Data', id='load-process-button', n_clicks=0),
        html.Div(id='processing-status')
    ]),
    dcc.Store(id='processed-data-store'),
    dcc.Tabs([
        dcc.Tab(label="Time Series Analysis", children=[
            dcc.Graph(id='time-series-graph'),
            dcc.Dropdown(
                id='feature-dropdown',
                options=[{'label': col, 'value': col} for col in ['Nitrate_Level', 'Phosphate_Level', 'Flow_Rate']],
                value='Nitrate_Level',
                style={'width': '50%'}
            )
        ]),
        dcc.Tab(label="Feature Importance", children=[
            dcc.Graph(id='feature-importance-graph')
        ]),
        dcc.Tab(label="Correlation Heatmap", children=[
            dcc.Graph(id='correlation-heatmap')
        ]),
        dcc.Tab(label="Model Predictions", children=[
            dcc.Graph(id='model-predictions-graph'),
            dcc.Dropdown(
                id='model-dropdown',
                options=[
                    {'label': 'Linear Regression', 'value': 'linear'},
                    {'label': 'Random Forest', 'value': 'rf'}
                ],
                value='linear',
                style={'width': '50%'}
            )
        ])
    ])
])

def load_and_process_data():
    # Load data
    sensor_data = pd.read_csv('sensor_data.csv', parse_dates=['Timestamp'])
    weather_data = pd.read_csv('weather_data.csv', parse_dates=['Timestamp'])
    
    # Merge data
    data = pd.merge(sensor_data, weather_data, on='Timestamp')
    
    # Basic feature engineering
    data['Hour'] = data['Timestamp'].dt.hour
    data['Day'] = data['Timestamp'].dt.day
    
    # Prepare features and target
    target = 'Nitrate_Level'
    features = [col for col in data.columns if col not in ['Timestamp', 'Sensor_ID', target]]
    X = data[features]
    y = data[target]
    
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    y_imputed = imputer.fit_transform(y.values.reshape(-1, 1)).ravel()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # Train models
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_imputed, test_size=0.2, random_state=42)
    
    linear_model = LinearRegression().fit(X_train, y_train)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
    
    # Calculate feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Convert Timestamp to string in ISO format
    data['Timestamp'] = data['Timestamp'].apply(lambda x: x.isoformat())

    return {
        'data': json.loads(data.to_json(orient='records', date_format='iso')),
        'features': features,
        'target': target,
        'X_scaled': X_scaled.tolist(),
        'y': y_imputed.tolist(),
        'feature_importance': feature_importance.to_dict('records'),
        'linear_model': None,  # We can't serialize sklearn models
        'rf_model': None  # We can't serialize sklearn models
    }

@app.callback(
    [Output('processed-data-store', 'data'),
     Output('processing-status', 'children')],
    [Input('load-process-button', 'n_clicks')]
)
def load_and_process(n_clicks):
    if n_clicks > 0:
        try:
            processed_data = load_and_process_data()
            return json.dumps(processed_data), 'Data loaded and processed successfully!'
        except Exception as e:
            logging.error(f"Error processing data: {str(e)}")
            return None, f'Error processing data: {str(e)}'
    return None, ''

@app.callback(
    Output('time-series-graph', 'figure'),
    [Input('feature-dropdown', 'value'),
     Input('processed-data-store', 'data')]
)
def update_time_series(feature, processed_data):
    if processed_data is None:
        return go.Figure()

    # Parse the JSON string into a Python dictionary
    processed_data = json.loads(processed_data)
    
    data = pd.DataFrame(processed_data['data'])
    fig = go.Figure()
    for sensor in data['Sensor_ID'].unique():
        sensor_data = data[data['Sensor_ID'] == sensor]
        fig.add_trace(go.Scatter(x=sensor_data['Timestamp'], y=sensor_data[feature], name=f'Sensor {sensor}'))
    
    if feature == 'Nitrate_Level':
        fig.add_hline(y=LOW_THRESHOLD, line_dash="dash", line_color="yellow", annotation_text="Low Threshold")
        fig.add_hline(y=HIGH_THRESHOLD, line_dash="dash", line_color="red", annotation_text="High Threshold")
    
    fig.update_layout(title=f'{feature} Over Time', xaxis_title='Timestamp', yaxis_title=feature)
    return fig

@app.callback(
    Output('feature-importance-graph', 'figure'),
    [Input('processed-data-store', 'data')]
)
def update_feature_importance(processed_data):
    if processed_data is None:
        return go.Figure()

    processed_data = json.loads(processed_data)
    feature_importance = pd.DataFrame(processed_data['feature_importance'])
    fig = px.bar(feature_importance.head(20), x='importance', y='feature', orientation='h')
    fig.update_layout(title='Top 20 Features by Importance', yaxis={'categoryorder':'total ascending'})
    return fig

@app.callback(
    Output('correlation-heatmap', 'figure'),
    [Input('processed-data-store', 'data')]
)
def update_correlation_heatmap(processed_data):
    if processed_data is None:
        return go.Figure()

    processed_data = json.loads(processed_data)
    data = pd.DataFrame(processed_data['data'])
    corr = data[processed_data['features'] + [processed_data['target']]].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdBu',
        zmin=-1, zmax=1
    ))
    fig.update_layout(title='Feature Correlation Heatmap')
    return fig

@app.callback(
    Output('model-predictions-graph', 'figure'),
    [Input('model-dropdown', 'value'),
     Input('processed-data-store', 'data')]
)
def update_model_predictions(model_type, processed_data):
    if processed_data is None:
        return go.Figure()

    processed_data = json.loads(processed_data)
    X_scaled = np.array(processed_data['X_scaled'])
    y = np.array(processed_data['y'])
    
    if model_type == 'linear':
        model = LinearRegression().fit(X_scaled, y)
        title = 'Linear Regression Predictions vs Actual'
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_scaled, y)
        title = 'Random Forest Predictions vs Actual'
    
    y_pred = model.predict(X_scaled)
    
    fig = go.Figure()
    timestamps = [datetime.fromisoformat(d['Timestamp']) for d in processed_data['data']]
    fig.add_trace(go.Scatter(x=timestamps, y=y, mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=timestamps, y=y_pred, mode='lines', name='Predicted'))
    fig.add_hline(y=LOW_THRESHOLD, line_dash="dash", line_color="yellow", annotation_text="Low Threshold")
    fig.add_hline(y=HIGH_THRESHOLD, line_dash="dash", line_color="red", annotation_text="High Threshold")
    
    fig.update_layout(title=title, xaxis_title='Timestamp', yaxis_title='Nitrate Level')
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)