# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.linear_model import LinearRegression

# %%
data = pd.read_csv("2) Simple Linear with sklearn - Exercise/real_estate_price_size.csv")
data.head()

# %%
data.describe()

# %%
x = data['size']
y = data['price']

# %%
plt.scatter(x,y)
plt.xlabel("size")
plt.ylabel("price")

# %%
x_matrix = x.values.reshape(-1, 1)

# %%
reg = LinearRegression()

# %%
reg.fit(x_matrix, y)

# %%
reg.score(x_matrix, y)

# %%
reg.intercept_

# %%
reg.coef_

# %%
value = [[750]]

reg.predict(value)

# %%
x_test = pd.DataFrame(data=[750], columns=['size'])
x_test['predict_price'] = reg.predict(x_test['size'].values.reshape(-1, 1))
x_test

# %%
plt.scatter(x, y)
yhat = reg.intercept_ + reg.coef_*x
plt.plot(x, yhat, color='red', linewidth=2)
plt.xlabel('size')
plt.ylabel('price')

# %%



