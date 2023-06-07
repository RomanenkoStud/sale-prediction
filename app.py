import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

# Load the dataset from CSV
dataset = pd.read_csv('database.csv')

# Convert the 'Month' column to datetime format
dataset['Month'] = pd.to_datetime(dataset['Month'])

# Encode the city column
label_encoder = LabelEncoder()
dataset['City'] = label_encoder.fit_transform(dataset['City'])

# Specify the target city and month for prediction
target_cities = ['London', 'New York', 'Kharkiv', 'Tokyo']
target_month = pd.to_datetime('2022-12-01')

# Get the features (city and month) and target variable (orders)
X = dataset[['City']].values
X_month = dataset['Month'].dt.month.values.reshape(-1, 1)
X = np.hstack((X, X_month))
y = dataset['Orders'].values.reshape(-1, 1)

# Normalize the features and target variable
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X.astype('float64'))
y_scaled = scaler_y.fit_transform(y.astype('float64'))

# Create a Linear Regression model
regression_model = LinearRegression()

# Fit the regression model
regression_model.fit(X_scaled, y_scaled)

# Create a Gradient Boosting model
gbr_model = GradientBoostingRegressor()

# Fit the Gradient Boosting model
gbr_model.fit(X_scaled, y_scaled)

# Build the neural network model
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X_scaled.shape[1]))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='tanh'))

model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.0001))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(X_scaled, y_scaled, epochs=100, batch_size=32, verbose=1)

# Iterate over the target cities
for target_city in target_cities:
    # Filter the dataset for the target city
    filtered_data = dataset[dataset['City'] == label_encoder.transform([target_city])[0]]

    # Sort the dataset by month in ascending order
    filtered_data = filtered_data.sort_values('Month')

    # Prepare the features for the target month
    target_city_encoded = label_encoder.transform([target_city])[0]

    # Loop to predict values from the next month to target month
    last_month = filtered_data['Month'].values[-1]
    months = pd.date_range(start=last_month + pd.DateOffset(months=1), end=target_month, freq='MS')
    predicted_sales = []
    regression_predicted_sales = []
    gbr_predicted_sales = []
    for month in months:
        # Predict the sales using model
        X_pred = np.array([[target_city_encoded, month.month]])
        X_pred_scaled = scaler_X.transform(X_pred.astype('float64'))
        pred_scaled = model.predict(X_pred_scaled)
        pred = scaler_y.inverse_transform(pred_scaled)
        predicted_sales.append(pred[0][0])
        
        # Predict the sales using linear regression
        regression_pred_scaled = regression_model.predict(X_pred_scaled)
        regression_pred = scaler_y.inverse_transform(regression_pred_scaled)
        regression_predicted_sales.append(regression_pred[0][0])

        # Predict the sales using gradient boosting
        gbr_pred_scaled = gbr_model.predict(X_pred_scaled)
        gbr_pred = scaler_y.inverse_transform(gbr_pred_scaled.reshape(-1, 1))
        gbr_predicted_sales.append(gbr_pred[0][0])

    # Plot observed and predicted sales
    plt.figure(figsize=(10, 6))
    observed_sales = filtered_data['Orders'].values
    predicted_sales = np.array(predicted_sales)
    regression_predicted_sales = np.array(regression_predicted_sales)
    gbr_predicted_sales = np.array(gbr_predicted_sales)
    all_sales = np.concatenate((observed_sales, predicted_sales))
    all_sales_with_regression = np.concatenate((observed_sales, regression_predicted_sales))
    all_sales_with_gbr = np.concatenate((observed_sales, gbr_predicted_sales))
    all_months = pd.date_range(start=filtered_data['Month'].values[0], periods=len(all_sales), freq='MS')

    plt.plot(all_months[:len(observed_sales)], all_sales[:len(observed_sales)], marker='o', label='Observed Sales')
    plt.plot(all_months[len(observed_sales)-1:], all_sales[len(observed_sales)-1:], marker='o', color='r', label='Custom predicted Sales')
    plt.plot(all_months[len(observed_sales)-1:], all_sales_with_regression[len(observed_sales)-1:], marker='o', color='m', label='Linear Regression')
    plt.plot(all_months[len(observed_sales)-1:], all_sales_with_gbr[len(observed_sales)-1:], marker='o', color='y', label='Gradient Boosting Regression')
    plt.xlabel('Month')
    plt.ylabel('Sales')
    plt.title(f'Sales Prediction for {target_city}')
    plt.legend()
    plt.grid(True)

plt.show()