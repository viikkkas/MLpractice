from sklearn import neighbors
from sklearn import datasets

knn = neighbors.KNeighborsClassifier(n_neighbors=10)
iris = datasets.load_iris()
X,y = iris.data,iris.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

knn.fit(X_train,y_train) 
#accuracy score = 0.95 

y_pred = knn.predict(X_test)
