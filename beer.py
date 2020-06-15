import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
df=pd.read_csv('C:/Users/Vikas/Downloads/Consumo_cerveja.csv')
data=df
x=df.loc[:,'Temperatura Media (C)':'Temperatura Minima (C)']
for i in set(x['Temperatura Media (C)']):
    x['Temperatura Media (C)']=x['Temperatura Media (C)'].replace(',','.')
    
for i in set(x['Temperatura Media (C)']):
    x['Temperatura Media (C)']=x['Temperatura Media (C)'].replace(',','.')
    
y=df['Consumo de cerveja (litros)']
C=1.0
svc=svm.SVC(kernel='rbf', C=1, gamma=100).fit(x,y)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
plt.subplot(1, 1, 1)
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.title('SVC with linear kernel')
plt.show()
