#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv ('regrex1.csv')

x = data [['x']]
y = data ['y']

plt.figure(figsize=(10, 8))
plt.scatter(data['x'], data['y'], color='cornflowerblue')

plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Scatter Plot Using Python')

plt.savefig('Scatterplot_Python.png')
plt.close()

plt.figure(figsize=(10, 8))
plt.scatter(data['x'], data['y'], color='cornflowerblue')

plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Scatter Plot with Linear Regression Using Python')

model = LinearRegression()
model.fit(x, y)

y_pred = model.predict(x)

plt.plot(data['x'], y_pred, color='maroon')

plt.savefig('Scatterplot_LR_Python.png')
plt.show()


# In[ ]:
