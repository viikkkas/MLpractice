import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
df=pd.read_csv('C:/Users/Vikas/Downloads/Housing_Data.csv')
reg=LinearRegression()
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
y=df['price']
x=df['lotsize']
y=pd.np.array(y)
x=pd.np.array(x)
y=y.reshape(-1,1)
x=x.reshape(-1,1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
reg.fit(x_train,y_train)
reg.predict(x_test)
plt.scatter(x,y)
plt.plot(x_test,reg.predict(x_test),color='red')
plt.show()
