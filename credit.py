from sklearn import DesicionTreeClassifier
import pandas as pd
df=pd.read_csv('C:/Users/Vikas/Downloads/DataSets/creditcard.csv')
data=df
data=data.drop('Class',axis=1)
target=df['Class']
