import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=4)
lr=LinearRegression()
df=pd.read_csv('C:/Users/Vikas/Downloads/Housing_Data.csv')
import matplotlib.pyplot as plt
x=(pd.np.array(df['lotsize'])).reshape(-1,1)
y=(pd.np.array(df['price'])).reshape(-1,1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
l=poly.fit_transform(x_train)
poly.fit(l,y_train)
lr.fit(l,y_train)
plt.scatter(x,y)
plt.plot(x_test,lr.predict(poly.fit_transform(x_test)),color='red')
plt.show()
