# model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# Load dataset once (instead of loading every request)
Data = pd.read_csv('/home/prabhakar/PycharmProject/House_Price_Prediction/housing.xls').dropna()

# Features and target
X = Data[['Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 'Avg. Area House Age',
          'Avg. Area Income', 'Area Population']]
y = Data['Price']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)
linear_pred_test = linear_model.predict(X_test_scaled)
linear_r2 = r2_score(y_test, linear_pred_test)

tree_model = DecisionTreeRegressor()
tree_model.fit(X_train, y_train)
tree_pred_test = tree_model.predict(X_test)
tree_r2 = r2_score(y_test, tree_pred_test)

forest_model = RandomForestRegressor(n_estimators=100)
forest_model.fit(X_train, y_train)
forest_pred_test = forest_model.predict(X_test)
forest_r2 = r2_score(y_test, forest_pred_test)

# Store models and metrics in a dictionary
models = {
    'Linear': {'model': linear_model, 'r2': linear_r2, 'scaled': True},
    'DecisionTree': {'model': tree_model, 'r2': tree_r2, 'scaled': False},
    'RandomForest': {'model': forest_model, 'r2': forest_r2, 'scaled': False}
}


def predict_best_model(NumberOfRooms, NumberOfBedrooms, HouseAge, AvgAreaIncome, AreaPopulation):
    user_input = [[NumberOfRooms, NumberOfBedrooms, HouseAge, AvgAreaIncome, AreaPopulation]]

    best_model_name = max(models, key=lambda x: models[x]['r2'])
    model_info = models[best_model_name]

    if model_info['scaled']:
        from sklearn.preprocessing import StandardScaler
        # scale user input
        user_input_scaled = scaler.transform(user_input)
        prediction = model_info['model'].predict(user_input_scaled)
    else:
        prediction = model_info['model'].predict(user_input)

    return float(prediction[0]), best_model_name
