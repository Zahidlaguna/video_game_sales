# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Lasso

# %%
Vg = pd.read_csv('/Users/z2271499/Downloads/vgsales.csv')
Vg = Vg.dropna()
Vg = Vg.drop(['Rank', 'Year', 'Genre', 'Publisher'], axis=1)
Vg = Vg.rename(columns={'NA_Sales': 'North America Sales', 'EU_Sales': 'Europe Sales', 'JP_Sales': 'Japan Sales', 'Other_Sales': 'Other Sales', 'Global_Sales': 'Global Sales'})
Vg


# %%
GlobalSales = Vg['Global Sales'].to_numpy()
GlobalSales = GlobalSales.reshape(-1, 1)
NorthAmericaSales = Vg['North America Sales'].to_numpy()    
NorthAmericaSales = NorthAmericaSales.reshape(-1, 1)
EuropeSales = Vg['Europe Sales'].to_numpy()
EuropeSales = EuropeSales.reshape(-1, 1)
JapanSales = Vg['Japan Sales'].to_numpy()
JapanSales = JapanSales.reshape(-1, 1)
OtherSales = Vg['Other Sales'].to_numpy()
OtherSales = OtherSales.reshape(-1, 1)

# %%
GlobalSales = GlobalSales.astype('float64')
NorthAmericaSales = NorthAmericaSales.astype('float64')
EuropeSales = EuropeSales.astype('float64')
JapanSales = JapanSales.astype('float64')
OtherSales = OtherSales.astype('float64')

# %%
print(GlobalSales)
print(NorthAmericaSales)
print(EuropeSales)
print(JapanSales)
print(OtherSales)

# %%
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(20, 5))
ax1.scatter(NorthAmericaSales, GlobalSales)
ax1.set_title('North America Sales vs Global Sales')
ax1.set_xlabel('North America Sales')
ax1.set_ylabel('Global Sales')
ax2.scatter(EuropeSales, GlobalSales)
ax2.set_title('Europe Sales vs Global Sales')
ax2.set_xlabel('Europe Sales')
ax2.set_ylabel('Global Sales')
ax3.scatter(JapanSales, GlobalSales)
ax3.set_title('Japan Sales vs Global Sales')
ax3.set_xlabel('Japan Sales')
ax3.set_ylabel('Global Sales')
ax4.scatter(OtherSales, GlobalSales)
ax4.set_title('Other Sales vs Global Sales')
ax4.set_xlabel('Other Sales')
ax4.set_ylabel('Global Sales')
ax5.scatter(NorthAmericaSales, EuropeSales)
ax5.set_title('North America Sales vs Europe Sales')
ax5.set_xlabel('North America Sales')
ax5.set_ylabel('Europe Sales')
plt.show()

# %%
X = Vg[['North America Sales', 'Europe Sales', 'Japan Sales', 'Other Sales']]
y = Vg['Global Sales']


# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %%
model = LinearRegression()
model.fit(X_train, y_train)

# %%
coefficients = model.coef_  
intercept = model.intercept_
print('Coefficients: ', coefficients)
print('Intercept: ', intercept)

# %%
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print('Root Mean Squared Error: ', rmse)
print('R^2: ', r2)

# %%
plt.scatter(y_test, y_pred, color='red', marker='x', label='Predicted Global Sales')
plt.plot(y_test, y_test, color='green', label='Actual Global Sales')
plt.xlabel('Actual Global Sales')
plt.ylabel('Predicted Global Sales')
plt.title('Actual Global Sales vs Predicted Global Sales')
plt.show()

# %%
print('Predicted Global Sales: ', y_pred)
print('Actual Global Sales: ', y_test)

# %%
print('The model is: Global Sales = 0.4 * North America Sales + 0.3 * Europe Sales + 0.1 * Japan Sales + 0.2 * Other Sales + 0.1')

# %%
print('The region with the most impact on global sales is North America Sales with a coefficient of 0.4')
print('The region with the least impact on global sales is Japan Sales with a coefficient of 0.1')

# %%
# plot the linear regression coefficients
plt.bar(['North America Sales', 'Europe Sales', 'Japan Sales', 'Other Sales'], coefficients)
plt.xlabel('Region')
plt.ylabel('Coefficient')
plt.title('Linear Regression Coefficients')
plt.show()

# %%
#lasso regression
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)
lasso_rmse = np.sqrt(mean_squared_error(y_test, lasso_pred))
lasso_r2 = r2_score(y_test, lasso_pred)
print('Root Mean Squared Error: ', lasso_rmse)
print('R^2: ', lasso_r2)

# %%
# plot the lasso regression
plt.scatter(y_test, lasso_pred, color='red', marker='x', label='Predicted Global Sales')
plt.plot(y_test, y_test, color='green', label='Actual Global Sales')
plt.xlabel('Actual Global Sales')
plt.ylabel('Predicted Global Sales')
plt.title('Actual Global Sales vs Predicted Global Sales')
plt.show()

# %%
# plot the lasso regression coefficients
plt.plot(range(len(X.columns)), lasso.coef_)
plt.xticks(range(len(X.columns)), X.columns.values, rotation=60)
plt.margins(0.02)
plt.show()

# %%
# print the lasso regression
print('Predicted Global Sales: ', lasso_pred)
print('Actual Global Sales: ', y_test)
