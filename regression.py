import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from keras import Sequential
from keras import layers
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Įkeliam duomenų rinkinį
data = pd.read_csv('realtor-data.csv')

# Peržiūrime pirmas eilutes, kad suprastume duomenų struktūrą
print(data.head())

# Patikriname duomenų tipus ir trūkstamus reikšmes
print(data.info())

# Aprašome pagrindines statistikas
print(data.describe())



#pasiimame reikalingus stulpelius
data = data[[ 'price', 'bed', 'bath', 'acre_lot', 'street', 'city', 'state', 'zip_code', 'house_size']]

# Pašaliname eilutes su trūkstamomis reikšmėmis
data.dropna(inplace=True)

# Koduojame kategorinius kintamuosius
label_encoder = LabelEncoder()
#data['status'] = label_encoder.fit_transform(data['status'])
data['city'] = label_encoder.fit_transform(data['city'])
data['state'] = label_encoder.fit_transform(data['state'])
#data['prev_sold_date'] = label_encoder.fit_transform(data['prev_sold_date'])

# Standartizuojame skaitinius požymius
scaler = StandardScaler()
numerical_features = ['price', 'bed', 'bath', 'acre_lot', 'street', 'zip_code', 'house_size'] # pakeiskite šiuos požymius pagal savo duomenų rinkinį
data[numerical_features] = scaler.fit_transform(data[numerical_features])

print(data.head())

# Padaliname duomenis į mokymo ir testavimo rinkinius
X = data.drop(['price'], axis=1)
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Tiesinė regresija
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
linear_reg_predictions = linear_reg.predict(X_test)

# Polinominė regresija
poly_features = PolynomialFeatures(degree=2)  # pasirinkite tinkamą polinomo laipsnį
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)
poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)
poly_reg_predictions = poly_reg.predict(X_test_poly)

# Decision Tree Regression
decision_tree = DecisionTreeRegressor(random_state=42)  # Initialize DecisionTreeRegressor
decision_tree.fit(X_train, y_train)  # Fit decision tree model
decision_tree_predictions = decision_tree.predict(X_test)  # Make predictions

# RNN
model = Sequential([
    layers.SimpleRNN(units=32, input_shape=(X_train.shape[1], 1), activation='relu', return_sequences=True),
    layers.Dropout(0.2),
    layers.SimpleRNN(units=32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(units=1)
])

# Modelio kompiliavimas
model.compile(optimizer='adam', loss='mean_absolute_error')

# Modelio apmokymas
model.fit(X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1)), y_train, epochs=15, batch_size=32)

# Modelio prognozės
rnn_predictions = model.predict(X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1)))

# Įvertiname modelių našumą
linear_reg_mae = mean_absolute_error(y_test, linear_reg_predictions)
linear_reg_mse = mean_squared_error(y_test, linear_reg_predictions)
poly_reg_mae = mean_absolute_error(y_test, poly_reg_predictions)
poly_reg_mse = mean_squared_error(y_test, poly_reg_predictions)
decision_tree_mae = mean_absolute_error(y_test, decision_tree_predictions)
decision_tree_mse = mean_squared_error(y_test, decision_tree_predictions)
rnn_mae = mean_absolute_error(y_test, rnn_predictions)
rnn_mse = mean_squared_error(y_test, rnn_predictions)

print("Tiesinės regresijos MAE:", linear_reg_mae)
print("Tiesinės regresijos MSE:", linear_reg_mse)
print("Polinominės regresijos MAE:", poly_reg_mae)
print("Polinominės regresijos MSE:", poly_reg_mse)
print("Decision Tree Regression MAE:", decision_tree_mae)
print("Decision Tree Regression MSE:", decision_tree_mse)
print("RNN MAE:", rnn_mae)
print("RNN MSE:", rnn_mse)

# Palyginimo grafikas
plt.figure(figsize=(10, 6))

# Tiesinė regresija
plt.plot(y_test, linear_reg_predictions, 'o', label='Linear Regression')

# Polinominė regresija
plt.plot(y_test, poly_reg_predictions, 'o', label='Polynomial Regression')

# decision trees
plt.plot(y_test, decision_tree_predictions, 'o', label='Decision Tree Regression')  # Plot decision tree predictions

# RNN
plt.plot(y_test, rnn_predictions, 'o', label='RNN')

plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Model Prediction Comparison')
plt.legend()
plt.grid(True)
plt.show()