# Package import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Datasets imports
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# Ploting prices and log prices 
fig = plt.figure(figsize=(10,5))
plt.subplot(121)
plt.title('Prices distribution')
plt.hist(train['SalePrice'], bins=50)
plt.ylabel('Sale Price')
plt.subplot(122)
plt.title('Log prices distribution')
plt.hist(np.log(train['SalePrice']), bins=50)
plt.ylabel('Sale Price (log)')
plt.tight_layout()
plt.show()

# Is price distribution normal?
print('Shapiro Wilks test (raw) : '+str(sp.stats.shapiro(train['SalePrice'].values)))
print('Shapiro Wilks test (log) : '+str(sp.stats.shapiro(np.log(train['SalePrice'].values))))

# Ploting a QQ plot
fig = plt.figure(figsize=(10,5))
plt.subplot(121)
sp.stats.probplot((train['SalePrice']), dist="norm", plot=plt)
plt.title('Positively skewed distribution')
plt.subplot(122)
sp.stats.probplot(np.log(train['SalePrice']), dist='norm', plot=plt)
plt.title('Normal distribution')
plt.tight_layout()
plt.show()


# Linear regression models
fig = plt.figure(figsize=(12,7))
plt.subplot(221)
plt.title('Log distribution of SalePrice')
plt.scatter(train['GrLivArea'], np.log(train['SalePrice']), c='#98d3cd', alpha=0.4)

# Features used
features = ["GrLivArea", "MSSubClass", 'LotArea', 'TotRmsAbvGrd', 'OverallQual', 'YearBuilt','YrSold']

# Train and testing sets
X_train, X_test, y_train, y_test = train_test_split(train[features], np.log(train["SalePrice"]), test_size=0.2, random_state = 42)

# Linear regression model
linear = LinearRegression()

# Training the model with only GrLivArea
X_train_1D = X_train["GrLivArea"].values.reshape(-1,1)
linear.fit(X_train_1D, y_train)
print("First model (GrLivArea only)")
print("Accuracy = "+str(1-linear.score(X_test["GrLivArea"].values.reshape(-1,1), y_test)))

# Plotting the first model
plt.subplot(222)
plt.title('First model (GrLivArea only)')
plt.scatter(train["GrLivArea"], np.log(train["SalePrice"]), c='#98d3cd', alpha=0.4)
plt.plot(X_train_1D, linear.predict(X_train_1D), c="#777777")

# Second model with all the features
linear.fit(X_train, y_train)
print("Second model")
print("Error = "+str(1-linear.score(X_test, y_test)))

# Graph with the model with all the features
plt.subplot(223)
plt.title('Second model (all features)')
plt.scatter(train["GrLivArea"], np.log(train["SalePrice"]), c='#98d3cd')
plt.scatter(X_train_1D, linear.predict(X_train), c="#777777", s=5)

# Third model 
poly = PolynomialFeatures(2)
X_train2 = poly.fit_transform(X_train)
X_score2 = poly.fit_transform(X_test)
linear.fit(X_train2, y_train)
print("Third model")
print("Error = "+str(1-linear.score(X_score2, y_test)))

# Plot
plt.subplot(224)
plt.title('Third model (all features, 2nd degree polynom)')
plt.scatter(train["GrLivArea"], np.log(train["SalePrice"]), c='#98d3cd')
plt.scatter(X_train_1D, linear.predict(X_train2), c="#777777", s=5)
plt.tight_layout()
plt.show()


# Gradient boosting model
from sklearn.ensemble import GradientBoostingRegressor

# Fitting the model
gradient = GradientBoostingRegressor(n_estimators=100)
gradient.fit(X_train2, y_train)
print("Gradient Boost")
print("Error = "+str(1-gradient.score(X_score2, y_test)))

# Ploting the model
plt.title('Gradient boosting regressor (all features)')
plt.scatter(train["GrLivArea"], np.log(train["SalePrice"]), c='#98d3cd')
plt.scatter(X_train_1D, gradient.predict(X_train2), c="#777777", s=5)
plt.show()

# TKaggle predictions
result = gradient.predict(poly.fit_transform(test[features]))
result = np.exp(result)
sub = pd.DataFrame({'Id':test['Id'], 'SalePrice':result})
sub.to_csv('Submission.csv', index=False)