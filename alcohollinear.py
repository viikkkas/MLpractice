import pandas as pd
import numpy as np
df=pd.read_csv('C:/Users/Vikas/Downloads/Consumo_cerveja.csv')
x=np.array(df.iloc[:,[1,2]])
y=np.array(df['Consumo de cerveja (litros)'])
##x=x.reshape(-1,1)
##y=y.reshape(-1,1)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
reg=LinearRegression()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
reg.fit(x_train,y_train)
reg.predict(x_test)
plt.plot(x,y,'ro')
plt.plot(x_test,reg.predict(x_test),color='red')
plt.xlabel('Temperatura Media (C)')
plt.ylabel('Consumption (litres)')
plt.title('Beer Consumption')
plt.show()
