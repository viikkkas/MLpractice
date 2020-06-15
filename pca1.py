from sklearn.decomposition import PCA
import pandas as pd,numpy as np
data=pd.read_csv('C:/Users/Vikas/Downloads/Mall_Customers.csv')
data=data.drop('Gender',axis=1)
pca=PCA(2)
pca.fit(data)
d=pca.fit(data).transform(data)
from sklearn.cluster import KMeans
k=5
