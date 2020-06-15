Python 3.7.3 (v3.7.3:ef4ec6ed12, Mar 25 2019, 21:26:53) [MSC v.1916 32 bit (Intel)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> from sklearn import neighbors
>>> knn=neighbors.KNeighborsClassifier(n_neighbors=1)
>>> from sklearn import datasets
>>> iris=datasets.load_iris()
>>> ip,op=iris.data,iris.target
>>> knn.fit(ip,op)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=1, p=2,
                     weights='uniform')
>>> knn.score()
Traceback (most recent call last):
  File "<pyshell#6>", line 1, in <module>
    knn.score()
TypeError: score() missing 2 required positional arguments: 'X' and 'y'
>>> knn.score(ip,op)
1.0
>>> knn=neighbors.KNeighborsClassifier(n_neighbors=8)
>>> knn=neighbors.KNeighborsClassifier(n_neighbors=15)
>>> knn.fit(ip,op)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=15, p=2,
                     weights='uniform')
>>> knn.score(ip,op)
0.9866666666666667
>>> knn=neighbors.KNeighborsClassifier(n_neighbors=20)
>>> knn.fit(ip,op)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=20, p=2,
                     weights='uniform')
>>> knn.score(ip,op)
0.98
>>> knn=neighbors.KNeighborsClassifier(n_neighbors=18)
>>> knn.fit(ip,op)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=18, p=2,
                     weights='uniform')
>>> knn.score(ip,op)
0.9733333333333334
>>> knn=neighbors.KNeighborsClassifier(n_neighbors=16)
>>> knn.fit(ip,op)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=16, p=2,
                     weights='uniform')
>>> knn.score(ip,op)
0.9866666666666667
>>> 
