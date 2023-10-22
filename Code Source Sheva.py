import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import requests

# Mengunduh dan mengeksekusi konten dari file .py
url = 'https://raw.githubusercontent.com/ShevaUNM/Regresi-Linear/main/data%20sheva.py'
response = requests.get(url)
exec(response.text)

# Konversi data ke DataFrame
df = pd.DataFrame(data)

# Memisahkan fitur (X) dan target (Y)
X = df[['X1', 'X2', 'X3', 'X4', 'X5']]
y = df['Y']

# Membangun model regresi linier
model = LinearRegression()
model.fit(X, y)

# Menampilkan koefisien dan intercept dari model
print("Koefisien: ", model.coef_)
print("Intercept: ", model.intercept_)

# Menambahkan kolom konstanta (intercept)
df = sm.add_constant(df)

# Membentuk model regresi
model = sm.OLS(df['Y'], df[['const', 'X1', 'X2', 'X3', 'X4', 'X5']])
hasil = model.fit()

# Menampilkan persamaan regresi
print(hasil.summary())

# Membaca data dari URL
url = 'https://raw.githubusercontent.com/ShevaUNM/Regresi-Linear/main/data%20sheva.py'
response = requests.get(url)
exec(response.text)

# Memisahkan variabel
X1 = df['X1']
X2 = df['X2']
X3 = df['X3']
X4 = df['X4']
X5 = df['X5']
Y = df['Y']

# Membangun model regresi linier untuk setiap variabel
model_X1 = LinearRegression().fit(X1.values.reshape(-1, 1), Y)
model_X2 = LinearRegression().fit(X2.values.reshape(-1, 1), Y)
model_X3 = LinearRegression().fit(X3.values.reshape(-1, 1), Y)
model_X4 = LinearRegression().fit(X4.values.reshape(-1, 1), Y)
model_X5 = LinearRegression().fit(X5.values.reshape(-1, 1), Y)

# Prediksi menggunakan model
Y_pred_X1 = model_X1.predict(X1.values.reshape(-1, 1))
Y_pred_X2 = model_X2.predict(X2.values.reshape(-1, 1))
Y_pred_X3 = model_X3.predict(X3.values.reshape(-1, 1))
Y_pred_X4 = model_X4.predict(X4.values.reshape(-1, 1))
Y_pred_X5 = model_X5.predict(X5.values.reshape(-1, 1))

# Buat DataFrame yang berisi data prediksi
heatmap_data = pd.DataFrame({
    'X1': X1,
    'X2': X2,
    'X3': X3,
    'X4': X4,
    'X5': X5,
    'Y_pred_X1': Y_pred_X1,
    'Y_pred_X2': Y_pred_X2,
    'Y_pred_X3': Y_pred_X3,
    'Y_pred_X4': Y_pred_X4,
    'Y_pred_X5': Y_pred_X5
})

# Membuat plot 3D menjadi line plot
fig_line = go.Figure()

# Plot data sebenarnya sebagai titik
fig_line.add_trace(go.Scatter(x=heatmap_data['X1'], y=Y, mode='markers', name='Data Points X1'))
fig_line.add_trace(go.Scatter(x=heatmap_data['X2'], y=Y, mode='markers', name='Data Points X2'))
fig_line.add_trace(go.Scatter(x=heatmap_data['X3'], y=Y, mode='markers', name='Data Points X3'))
fig_line.add_trace(go.Scatter(x=heatmap_data['X4'], y=Y, mode='markers', name='Data Points X4'))
fig_line.add_trace(go.Scatter(x=heatmap_data['X5'], y=Y, mode='markers', name='Data Points X5'))

# Plot hasil prediksi sebagai garis
fig_line.add_trace(go.Scatter(x=heatmap_data['X1'], y=heatmap_data['Y_pred_X1'], mode='lines', name='Regression Line X1'))
fig_line.add_trace(go.Scatter(x=heatmap_data['X2'], y=heatmap_data['Y_pred_X2'], mode='lines', name='Regression Line X2'))
fig_line.add_trace(go.Scatter(x=heatmap_data['X3'], y=heatmap_data['Y_pred_X3'], mode='lines', name='Regression Line X3'))
fig_line.add_trace(go.Scatter(x=heatmap_data['X4'], y=heatmap_data['Y_pred_X4'], mode='lines', name='Regression Line X4'))
fig_line.add_trace(go.Scatter(x=heatmap_data['X5'], y=heatmap_data['Y_pred_X5'], mode='lines', name='Regression Line X5'))

fig_line.update_layout(
    title='Line Plot - Variabel X1 Hingga X5 vs Y',
    xaxis_title='Variabel X1-X5',
    yaxis_title='Y',
)

fig_line.show()

# Menampilkan heatmap
fig_heatmap = px.imshow(heatmap_data.corr())
fig_heatmap.update_layout(
    title='Heatmap of Correlations between X1-X5 and Predicted Y',
    xaxis_title='Features (X1-X5)',
    yaxis_title='Features and Predicted Y'
)
fig_heatmap.show()
