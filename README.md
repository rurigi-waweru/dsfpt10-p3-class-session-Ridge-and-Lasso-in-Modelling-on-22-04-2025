# Ridge and Lasso in Modelling

```python
import pandas as pd
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"
data = pd.read_csv(url)
df = data.dropna()
```

```python
# Display sample
print(df.head())
df.head()
df.head(2)
```

```python
# split the data
y = df.mpg
X = df.drop(['mpg', 'origin', 'name'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=12)
y.head()
```

```python
# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### further read: [click this link](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)


```python
# build ridge, lasso and reular linear regresson model
# In scikit, the regularization parameter is denoted by alpha
ridge = Ridge(alpha = 0.5)
ridge.fit(X_train_scaled, y_train)
```

```python
lasso = Lasso(alpha=0.5)
lasso.fit(X_train_scaled, y_train)
```

```python
lin = LinearRegression()
lin.fit(X_train_scaled, y_train)
# generate prediction for training and test sets
ridge_train = ridge.predict(X_train_scaled)
ridge_test = ridge.predict(X_test_scaled)
```


```python
lasso_train = lasso.predict(X_train_scaled)
lasso_test = lasso.predict(X_test_scaled)
```


```python
lin_train = lin.predict(X_train_scaled)
lin_test = lin.predict(X_test_scaled)
```

```python
# print the mean_swuared-error for train and test
print('Train error ridge:', mean_squared_error(y_train, ridge_train))
print('Test error ridge:', mean_squared_error(y_test, ridge_test))
print('\n')

print('Train error lasso:', mean_squared_error(y_train, lasso_train))
print('Test error lasso:', mean_squared_error(y_test, lasso_test))
print('\n')

print('Train error lin:', mean_squared_error(y_train, lin_train))
print('Test error lin:', mean_squared_error(y_test, lin_test))
# print('\n')
```


```python
# how including ridge and lasso chnaged our paraeter estimates
print('ridge parameter coeff:', ridge.coef_, end = '\n\n')
print('lasso parameter coeff:', lasso.coef_, end = '\n\n')
print('lin parameter coeff:', lin.coef_)
```