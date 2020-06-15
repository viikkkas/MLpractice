Python 3.7.3 (v3.7.3:ef4ec6ed12, Mar 25 2019, 21:26:53) [MSC v.1916 32 bit (Intel)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> from sklearn import datasets
>>> iris=datasets.load_iris
>>> iris=datasets.load_iris()
>>> ds=datasets.load_digits()
>>> ds.target_names
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> ds.feature_names
Traceback (most recent call last):
  File "C:\Users\Vikas\AppData\Local\Programs\Python\Python37-32\lib\site-packages\sklearn\utils\__init__.py", line 105, in __getattr__
    return self[key]
KeyError: 'feature_names'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<pyshell#5>", line 1, in <module>
    ds.feature_names
  File "C:\Users\Vikas\AppData\Local\Programs\Python\Python37-32\lib\site-packages\sklearn\utils\__init__.py", line 107, in __getattr__
    raise AttributeError(key)
AttributeError: feature_names
>>> ds.data
array([[ 0.,  0.,  5., ...,  0.,  0.,  0.],
       [ 0.,  0.,  0., ..., 10.,  0.,  0.],
       [ 0.,  0.,  0., ..., 16.,  9.,  0.],
       ...,
       [ 0.,  0.,  1., ...,  6.,  0.,  0.],
       [ 0.,  0.,  2., ..., 12.,  0.,  0.],
       [ 0.,  0., 10., ..., 12.,  1.,  0.]])
>>> ip,op=ds.data,ds.target
>>> from sklearn import neighbors
>>> knn=neighbors.KNeighborsClassifier()
>>> knn.fit(ip,op)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=5, p=2,
                     weights='uniform')
>>> knn.predict([1])
Traceback (most recent call last):
  File "<pyshell#11>", line 1, in <module>
    knn.predict([1])
  File "C:\Users\Vikas\AppData\Local\Programs\Python\Python37-32\lib\site-packages\sklearn\neighbors\classification.py", line 147, in predict
    X = check_array(X, accept_sparse='csr')
  File "C:\Users\Vikas\AppData\Local\Programs\Python\Python37-32\lib\site-packages\sklearn\utils\validation.py", line 521, in check_array
    "if it contains a single sample.".format(array))
ValueError: Expected 2D array, got 1D array instead:
array=[1].
Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
>>> knn.predict([[1]])
Traceback (most recent call last):
  File "<pyshell#12>", line 1, in <module>
    knn.predict([[1]])
  File "C:\Users\Vikas\AppData\Local\Programs\Python\Python37-32\lib\site-packages\sklearn\neighbors\classification.py", line 149, in predict
    neigh_dist, neigh_ind = self.kneighbors(X)
  File "C:\Users\Vikas\AppData\Local\Programs\Python\Python37-32\lib\site-packages\sklearn\neighbors\base.py", line 454, in kneighbors
    for s in gen_even_slices(X.shape[0], n_jobs)
  File "C:\Users\Vikas\AppData\Local\Programs\Python\Python37-32\lib\site-packages\joblib\parallel.py", line 921, in __call__
    if self.dispatch_one_batch(iterator):
  File "C:\Users\Vikas\AppData\Local\Programs\Python\Python37-32\lib\site-packages\joblib\parallel.py", line 759, in dispatch_one_batch
    self._dispatch(tasks)
  File "C:\Users\Vikas\AppData\Local\Programs\Python\Python37-32\lib\site-packages\joblib\parallel.py", line 716, in _dispatch
    job = self._backend.apply_async(batch, callback=cb)
  File "C:\Users\Vikas\AppData\Local\Programs\Python\Python37-32\lib\site-packages\joblib\_parallel_backends.py", line 182, in apply_async
    result = ImmediateResult(func)
  File "C:\Users\Vikas\AppData\Local\Programs\Python\Python37-32\lib\site-packages\joblib\_parallel_backends.py", line 549, in __init__
    self.results = batch()
  File "C:\Users\Vikas\AppData\Local\Programs\Python\Python37-32\lib\site-packages\joblib\parallel.py", line 225, in __call__
    for func, args, kwargs in self.items]
  File "C:\Users\Vikas\AppData\Local\Programs\Python\Python37-32\lib\site-packages\joblib\parallel.py", line 225, in <listcomp>
    for func, args, kwargs in self.items]
  File "C:\Users\Vikas\AppData\Local\Programs\Python\Python37-32\lib\site-packages\sklearn\neighbors\base.py", line 291, in _tree_query_parallel_helper
    return tree.query(data, n_neighbors, return_distance)
  File "sklearn\neighbors\binary_tree.pxi", line 1317, in sklearn.neighbors.kd_tree.BinaryTree.query
ValueError: query data dimension must match training data dimension
>>> knn.predict(arange(64))
Traceback (most recent call last):
  File "<pyshell#13>", line 1, in <module>
    knn.predict(arange(64))
NameError: name 'arange' is not defined
>>> knn.predict(np.arange(64))
Traceback (most recent call last):
  File "<pyshell#14>", line 1, in <module>
    knn.predict(np.arange(64))
NameError: name 'np' is not defined
>>> knn.predict(np.arange[64])
Traceback (most recent call last):
  File "<pyshell#15>", line 1, in <module>
    knn.predict(np.arange[64])
NameError: name 'np' is not defined
>>> import numpy as np
>>> knn.predict(np.arange[64])
Traceback (most recent call last):
  File "<pyshell#17>", line 1, in <module>
    knn.predict(np.arange[64])
TypeError: 'builtin_function_or_method' object is not subscriptable
>>> knn.predict(np.arange([64]))
Traceback (most recent call last):
  File "<pyshell#18>", line 1, in <module>
    knn.predict(np.arange([64]))
TypeError: unsupported operand type(s) for -: 'list' and 'int'
>>> knn.predict(np.arange([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]]))
Traceback (most recent call last):
  File "<pyshell#19>", line 1, in <module>
    knn.predict(np.arange([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]]))
TypeError: unsupported operand type(s) for -: 'list' and 'int'
>>> knn.predict([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])
array([0])
>>> ds.target_names(knn.predict([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]]))
Traceback (most recent call last):
  File "<pyshell#21>", line 1, in <module>
    ds.target_names(knn.predict([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]]))
TypeError: 'numpy.ndarray' object is not callable
>>> ds.target_names[knn.predict([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])]
array([0])
>>> knn.score(ip,op)
0.9905397885364496
>>> from sklearn.model_selection import train_test_split
>>> x_train,x_test,y_train,y_test=train_test_split(ds.data,ds.target,size=0.2)
Traceback (most recent call last):
  File "<pyshell#25>", line 1, in <module>
    x_train,x_test,y_train,y_test=train_test_split(ds.data,ds.target,size=0.2)
  File "C:\Users\Vikas\AppData\Local\Programs\Python\Python37-32\lib\site-packages\sklearn\model_selection\_split.py", line 2094, in train_test_split
    raise TypeError("Invalid parameters passed: %s" % str(options))
TypeError: Invalid parameters passed: {'size': 0.2}
>>> x_train,x_test,y_train,y_test=train_test_split(ds.data,ds.target,test_size=0.2)
>>> knn.predict(x_test)
array([8, 3, 6, 5, 7, 1, 1, 1, 1, 0, 4, 3, 0, 9, 6, 9, 6, 9, 4, 0, 8, 3,
       9, 9, 0, 3, 4, 2, 2, 0, 1, 9, 9, 3, 9, 6, 5, 2, 0, 6, 0, 2, 1, 1,
       4, 2, 2, 0, 9, 9, 7, 8, 4, 7, 3, 5, 3, 0, 2, 5, 2, 4, 9, 4, 8, 0,
       1, 2, 4, 5, 6, 9, 8, 2, 1, 9, 7, 7, 5, 0, 2, 3, 9, 7, 2, 3, 6, 5,
       9, 4, 6, 6, 5, 8, 4, 4, 3, 1, 6, 3, 8, 4, 9, 8, 9, 8, 9, 8, 9, 0,
       1, 5, 7, 4, 0, 5, 2, 7, 7, 9, 0, 8, 2, 2, 1, 3, 0, 9, 5, 7, 3, 7,
       4, 3, 7, 1, 6, 9, 5, 6, 7, 1, 8, 1, 8, 1, 3, 2, 0, 1, 2, 9, 5, 3,
       2, 2, 5, 1, 1, 4, 1, 8, 5, 5, 0, 3, 8, 5, 4, 8, 9, 6, 8, 3, 8, 9,
       4, 9, 0, 2, 5, 6, 8, 5, 3, 7, 3, 7, 5, 2, 9, 4, 5, 4, 5, 6, 8, 7,
       0, 4, 4, 0, 9, 1, 9, 5, 6, 5, 9, 5, 7, 2, 9, 7, 8, 5, 0, 3, 8, 6,
       2, 5, 6, 9, 5, 7, 4, 9, 9, 8, 1, 9, 0, 7, 5, 5, 4, 4, 9, 1, 7, 0,
       5, 5, 9, 6, 5, 1, 1, 7, 6, 6, 1, 6, 2, 0, 2, 3, 1, 0, 4, 9, 1, 9,
       1, 3, 8, 9, 8, 8, 5, 2, 5, 1, 8, 0, 5, 2, 4, 0, 0, 3, 5, 5, 8, 6,
       8, 6, 7, 5, 8, 1, 9, 9, 2, 0, 3, 7, 6, 2, 8, 7, 4, 2, 6, 3, 4, 2,
       9, 5, 6, 6, 4, 2, 5, 9, 4, 6, 2, 5, 1, 7, 0, 8, 6, 0, 7, 0, 9, 3,
       8, 2, 3, 0, 0, 4, 7, 2, 2, 6, 5, 1, 3, 7, 9, 5, 8, 5, 1, 7, 0, 4,
       9, 8, 7, 5, 2, 5, 1, 2])
>>> from sklearn.metrics import accuracy_score
>>> accuracy_score(y_test,knn.predict(x_test))
0.9888888888888889
>>> knn.fit(x_train,y_train)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=5, p=2,
                     weights='uniform')
>>> knn.score
<bound method ClassifierMixin.score of KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=5, p=2,
                     weights='uniform')>
>>> knn.score(x_train,y_train)
0.9902574808629089
>>> ds.data.shape
(1797, 64)
>>> import matplotlib.pyplot as plt
>>> plt.plot([1,2,3],[4,5,6])
[<matplotlib.lines.Line2D object at 0x10269330>]
>>> plt.show
<function show at 0x0E3EA228>
>>> plt.show()
>>> plt.plot([[1,2,3],[4,5,6]])
[<matplotlib.lines.Line2D object at 0x125277D0>, <matplotlib.lines.Line2D object at 0x125278B0>, <matplotlib.lines.Line2D object at 0x12527970>]
>>> plt.show()
>>> plt.plot([1,2,3],[4,5,6])
[<matplotlib.lines.Line2D object at 0x10029D50>]
>>> plt.show()
>>> plt.plot([[1,2,3],[4,5,6]])
[<matplotlib.lines.Line2D object at 0x1007C250>, <matplotlib.lines.Line2D object at 0x1007C330>, <matplotlib.lines.Line2D object at 0x1007C3F0>]
>>> plt.axes([0,6,0,10])
<matplotlib.axes._axes.Axes object at 0x1006EBD0>
>>> plt.show()
>>> plt.plot([1,2,3],[4,5,6])
[<matplotlib.lines.Line2D object at 0x100E3770>]
>>> plt.axes([0,6,0,10])
<matplotlib.axes._axes.Axes object at 0x100E3A10>
>>> plt.show()
>>> plt.axis([0,6,0,10])
[0, 6, 0, 10]
>>> plt.show()
>>> plt.plot([1,2,3],[4,5,6])
[<matplotlib.lines.Line2D object at 0x10208EB0>]
>>> plt.axis([0,6,0,10])
[0, 6, 0, 10]
>>> plt.show()
>>> plt.show()
>>> plt.plot([1,2,3],[4,5,6])
[<matplotlib.lines.Line2D object at 0x108685B0>]
>>> plt.axis([0,6,0,10])
[0, 6, 0, 10]
>>> plt.xlabel('x-axis')
Text(0.5, 0, 'x-axis')
>>> plt.ylabel('y-axis')
Text(0, 0.5, 'y-axis')
>>> plt.title('testing')
Text(0.5, 1.0, 'testing')
>>> plt.show()
>>> plt.plot([[1,2,3],[4,5,6]],'ro')
[<matplotlib.lines.Line2D object at 0x0FD7BEB0>, <matplotlib.lines.Line2D object at 0x0FD7BF90>, <matplotlib.lines.Line2D object at 0x0FD8A070>]
>>> plt.show()
>>> help(plt.plot)
Help on function plot in module matplotlib.pyplot:

plot(*args, scalex=True, scaley=True, data=None, **kwargs)
    Plot y versus x as lines and/or markers.
    
    Call signatures::
    
        plot([x], y, [fmt], *, data=None, **kwargs)
        plot([x], y, [fmt], [x2], y2, [fmt2], ..., **kwargs)
    
    The coordinates of the points or line nodes are given by *x*, *y*.
    
    The optional parameter *fmt* is a convenient way for defining basic
    formatting like color, marker and linestyle. It's a shortcut string
    notation described in the *Notes* section below.
    
    >>> plot(x, y)        # plot x and y using default line style and color
    >>> plot(x, y, 'bo')  # plot x and y using blue circle markers
    >>> plot(y)           # plot y using x as index array 0..N-1
    >>> plot(y, 'r+')     # ditto, but with red plusses
    
    You can use `.Line2D` properties as keyword arguments for more
    control on the appearance. Line properties and *fmt* can be mixed.
    The following two calls yield identical results:
    
    >>> plot(x, y, 'go--', linewidth=2, markersize=12)
    >>> plot(x, y, color='green', marker='o', linestyle='dashed',
    ...      linewidth=2, markersize=12)
    
    When conflicting with *fmt*, keyword arguments take precedence.
    
    
    **Plotting labelled data**
    
    There's a convenient way for plotting objects with labelled data (i.e.
    data that can be accessed by index ``obj['y']``). Instead of giving
    the data in *x* and *y*, you can provide the object in the *data*
    parameter and just give the labels for *x* and *y*::
    
    >>> plot('xlabel', 'ylabel', data=obj)
    
    All indexable objects are supported. This could e.g. be a `dict`, a
    `pandas.DataFame` or a structured numpy array.
    
    
    **Plotting multiple sets of data**
    
    There are various ways to plot multiple sets of data.
    
    - The most straight forward way is just to call `plot` multiple times.
      Example:
    
      >>> plot(x1, y1, 'bo')
      >>> plot(x2, y2, 'go')
    
    - Alternatively, if your data is already a 2d array, you can pass it
      directly to *x*, *y*. A separate data set will be drawn for every
      column.
    
      Example: an array ``a`` where the first column represents the *x*
      values and the other columns are the *y* columns::
    
      >>> plot(a[0], a[1:])
    
    - The third way is to specify multiple sets of *[x]*, *y*, *[fmt]*
      groups::
    
      >>> plot(x1, y1, 'g^', x2, y2, 'g-')
    
      In this case, any additional keyword argument applies to all
      datasets. Also this syntax cannot be combined with the *data*
      parameter.
    
    By default, each line is assigned a different style specified by a
    'style cycle'. The *fmt* and line property parameters are only
    necessary if you want explicit deviations from these defaults.
    Alternatively, you can also change the style cycle using the
    'axes.prop_cycle' rcParam.
    
    
    Parameters
    ----------
    x, y : array-like or scalar
        The horizontal / vertical coordinates of the data points.
        *x* values are optional and default to `range(len(y))`.
    
        Commonly, these parameters are 1D arrays.
    
        They can also be scalars, or two-dimensional (in that case, the
        columns represent separate data sets).
    
        These arguments cannot be passed as keywords.
    
    fmt : str, optional
        A format string, e.g. 'ro' for red circles. See the *Notes*
        section for a full description of the format strings.
    
        Format strings are just an abbreviation for quickly setting
        basic line properties. All of these and more can also be
        controlled by keyword arguments.
    
        This argument cannot be passed as keyword.
    
    data : indexable object, optional
        An object with labelled data. If given, provide the label names to
        plot in *x* and *y*.
    
        .. note::
            Technically there's a slight ambiguity in calls where the
            second label is a valid *fmt*. `plot('n', 'o', data=obj)`
            could be `plt(x, y)` or `plt(y, fmt)`. In such cases,
            the former interpretation is chosen, but a warning is issued.
            You may suppress the warning by adding an empty format string
            `plot('n', 'o', '', data=obj)`.
    
    Other Parameters
    ----------------
    scalex, scaley : bool, optional, default: True
        These parameters determined if the view limits are adapted to
        the data limits. The values are passed on to `autoscale_view`.
    
    **kwargs : `.Line2D` properties, optional
        *kwargs* are used to specify properties like a line label (for
        auto legends), linewidth, antialiasing, marker face color.
        Example::
    
        >>> plot([1,2,3], [1,2,3], 'go-', label='line 1', linewidth=2)
        >>> plot([1,2,3], [1,4,9], 'rs',  label='line 2')
    
        If you make multiple lines with one plot command, the kwargs
        apply to all those lines.
    
        Here is a list of available `.Line2D` properties:
    
      agg_filter: a filter function, which takes a (m, n, 3) float array and a dpi value, and returns a (m, n, 3) array
      alpha: float
      animated: bool
      antialiased or aa: bool
      clip_box: `.Bbox`
      clip_on: bool
      clip_path: [(`~matplotlib.path.Path`, `.Transform`) | `.Patch` | None]
      color or c: color
      contains: callable
      dash_capstyle: {'butt', 'round', 'projecting'}
      dash_joinstyle: {'miter', 'round', 'bevel'}
      dashes: sequence of floats (on/off ink in points) or (None, None)
      drawstyle or ds: {'default', 'steps', 'steps-pre', 'steps-mid', 'steps-post'}, default: 'default'
      figure: `.Figure`
      fillstyle: {'full', 'left', 'right', 'bottom', 'top', 'none'}
      gid: str
      in_layout: bool
      label: object
      linestyle or ls: {'-', '--', '-.', ':', '', (offset, on-off-seq), ...}
      linewidth or lw: float
      marker: marker style
      markeredgecolor or mec: color
      markeredgewidth or mew: float
      markerfacecolor or mfc: color
      markerfacecoloralt or mfcalt: color
      markersize or ms: float
      markevery: None or int or (int, int) or slice or List[int] or float or (float, float)
      path_effects: `.AbstractPathEffect`
      picker: float or callable[[Artist, Event], Tuple[bool, dict]]
      pickradius: float
      rasterized: bool or None
      sketch_params: (scale: float, length: float, randomness: float)
      snap: bool or None
      solid_capstyle: {'butt', 'round', 'projecting'}
      solid_joinstyle: {'miter', 'round', 'bevel'}
      transform: `matplotlib.transforms.Transform`
      url: str
      visible: bool
      xdata: 1D array
      ydata: 1D array
      zorder: float
    
    Returns
    -------
    lines
        A list of `.Line2D` objects representing the plotted data.
    
    See Also
    --------
    scatter : XY scatter plot with markers of varying size and/or color (
        sometimes also called bubble chart).
    
    Notes
    -----
    **Format Strings**
    
    A format string consists of a part for color, marker and line::
    
        fmt = '[marker][line][color]'
    
    Each of them is optional. If not provided, the value from the style
    cycle is used. Exception: If ``line`` is given, but no ``marker``,
    the data will be a line without markers.
    
    Other combinations such as ``[color][marker][line]`` are also
    supported, but note that their parsing may be ambiguous.
    
    **Markers**
    
    =============    ===============================
    character        description
    =============    ===============================
    ``'.'``          point marker
    ``','``          pixel marker
    ``'o'``          circle marker
    ``'v'``          triangle_down marker
    ``'^'``          triangle_up marker
    ``'<'``          triangle_left marker
    ``'>'``          triangle_right marker
    ``'1'``          tri_down marker
    ``'2'``          tri_up marker
    ``'3'``          tri_left marker
    ``'4'``          tri_right marker
    ``'s'``          square marker
    ``'p'``          pentagon marker
    ``'*'``          star marker
    ``'h'``          hexagon1 marker
    ``'H'``          hexagon2 marker
    ``'+'``          plus marker
    ``'x'``          x marker
    ``'D'``          diamond marker
    ``'d'``          thin_diamond marker
    ``'|'``          vline marker
    ``'_'``          hline marker
    =============    ===============================
    
    **Line Styles**
    
    =============    ===============================
    character        description
    =============    ===============================
    ``'-'``          solid line style
    ``'--'``         dashed line style
    ``'-.'``         dash-dot line style
    ``':'``          dotted line style
    =============    ===============================
    
    Example format strings::
    
        'b'    # blue markers with default shape
        'or'   # red circles
        '-g'   # green solid line
        '--'   # dashed line with default color
        '^k:'  # black triangle_up markers connected by a dotted line
    
    **Colors**
    
    The supported color abbreviations are the single letter codes
    
    =============    ===============================
    character        color
    =============    ===============================
    ``'b'``          blue
    ``'g'``          green
    ``'r'``          red
    ``'c'``          cyan
    ``'m'``          magenta
    ``'y'``          yellow
    ``'k'``          black
    ``'w'``          white
    =============    ===============================
    
    and the ``'CN'`` colors that index into the default property cycle.
    
    If the color is the only part of the format string, you can
    additionally use any  `matplotlib.colors` spec, e.g. full names
    (``'green'``) or hex strings (``'#008000'``).

>>> 
>>> plt.plot([[1,2,3],[4,5,6]],'or')
[<matplotlib.lines.Line2D object at 0x0FDC9370>, <matplotlib.lines.Line2D object at 0x0FD91250>, <matplotlib.lines.Line2D object at 0x0FDC94F0>]
>>> plt.show()
>>> plt.scatter([3,4,2],[8,6,2])
<matplotlib.collections.PathCollection object at 0x0FE068B0>
>>> plt.show()
>>> plt.imread('C:/Users/Vikas/Downloads/1494025719.pdf')
Traceback (most recent call last):
  File "C:\Users\Vikas\AppData\Local\Programs\Python\Python37-32\lib\site-packages\matplotlib\image.py", line 1405, in imread
    from PIL import Image
ModuleNotFoundError: No module named 'PIL'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<pyshell#68>", line 1, in <module>
    plt.imread('C:/Users/Vikas/Downloads/1494025719.pdf')
  File "C:\Users\Vikas\AppData\Local\Programs\Python\Python37-32\lib\site-packages\matplotlib\pyplot.py", line 2129, in imread
    return matplotlib.image.imread(fname, format)
  File "C:\Users\Vikas\AppData\Local\Programs\Python\Python37-32\lib\site-packages\matplotlib\image.py", line 1409, in imread
    'more images' % list(handlers))
ValueError: Only know how to handle extensions: ['png']; with Pillow installed matplotlib can handle more images
>>> plt.imread('C:/Users/Vikas/Downloads/937914.jpg')
Traceback (most recent call last):
  File "C:\Users\Vikas\AppData\Local\Programs\Python\Python37-32\lib\site-packages\matplotlib\image.py", line 1405, in imread
    from PIL import Image
ModuleNotFoundError: No module named 'PIL'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<pyshell#69>", line 1, in <module>
    plt.imread('C:/Users/Vikas/Downloads/937914.jpg')
  File "C:\Users\Vikas\AppData\Local\Programs\Python\Python37-32\lib\site-packages\matplotlib\pyplot.py", line 2129, in imread
    return matplotlib.image.imread(fname, format)
  File "C:\Users\Vikas\AppData\Local\Programs\Python\Python37-32\lib\site-packages\matplotlib\image.py", line 1409, in imread
    'more images' % list(handlers))
ValueError: Only know how to handle extensions: ['png']; with Pillow installed matplotlib can handle more images
>>> plt.imread('C:/Users/Vikas/Downloads/b31c4ced285e73f40c884d009c5a7da6.png')
array([[[1.        , 0.827451  , 0.01960784],
        [1.        , 0.827451  , 0.01960784],
        [1.        , 0.827451  , 0.01960784],
        ...,
        [1.        , 0.827451  , 0.01960784],
        [1.        , 0.827451  , 0.01960784],
        [1.        , 0.827451  , 0.01960784]],

       [[1.        , 0.827451  , 0.01960784],
        [1.        , 0.827451  , 0.01960784],
        [1.        , 0.827451  , 0.01960784],
        ...,
        [1.        , 0.827451  , 0.01960784],
        [1.        , 0.827451  , 0.01960784],
        [1.        , 0.827451  , 0.01960784]],

       [[1.        , 0.827451  , 0.01960784],
        [1.        , 0.827451  , 0.01960784],
        [1.        , 0.827451  , 0.01960784],
        ...,
        [1.        , 0.827451  , 0.01960784],
        [1.        , 0.827451  , 0.01960784],
        [1.        , 0.827451  , 0.01960784]],

       ...,

       [[1.        , 0.827451  , 0.01960784],
        [1.        , 0.827451  , 0.01960784],
        [1.        , 0.827451  , 0.01960784],
        ...,
        [1.        , 0.827451  , 0.01960784],
        [1.        , 0.827451  , 0.01960784],
        [1.        , 0.827451  , 0.01960784]],

       [[1.        , 0.827451  , 0.01960784],
        [1.        , 0.827451  , 0.01960784],
        [1.        , 0.827451  , 0.01960784],
        ...,
        [1.        , 0.827451  , 0.01960784],
        [1.        , 0.827451  , 0.01960784],
        [1.        , 0.827451  , 0.01960784]],

       [[1.        , 0.827451  , 0.01960784],
        [1.        , 0.827451  , 0.01960784],
        [1.        , 0.827451  , 0.01960784],
        ...,
        [1.        , 0.827451  , 0.01960784],
        [1.        , 0.827451  , 0.01960784],
        [1.        , 0.827451  , 0.01960784]]], dtype=float32)
>>> img=plt.imread('C:/Users/Vikas/Downloads/b31c4ced285e73f40c884d009c5a7da6.png')
>>> img=plt.imread('C:/Users/Vikas/Downloads/b31c4ced285e73f40c884d009c5a7da6.png')
>>> plt.imshow(img)
<matplotlib.image.AxesImage object at 0x0FDDB890>
>>> plt.show()
>>> img=plt.imread('C:/Users/Vikas/Downloads/002.png')
Traceback (most recent call last):
  File "<pyshell#75>", line 1, in <module>
    img=plt.imread('C:/Users/Vikas/Downloads/002.png')
  File "C:\Users\Vikas\AppData\Local\Programs\Python\Python37-32\lib\site-packages\matplotlib\pyplot.py", line 2129, in imread
    return matplotlib.image.imread(fname, format)
  File "C:\Users\Vikas\AppData\Local\Programs\Python\Python37-32\lib\site-packages\matplotlib\image.py", line 1426, in imread
    with open(fname, 'rb') as fd:
FileNotFoundError: [Errno 2] No such file or directory: 'C:/Users/Vikas/Downloads/002.png'
>>> img=plt.imread('C:/Users/Vikas/Downloads/002.png')
Traceback (most recent call last):
  File "<pyshell#76>", line 1, in <module>
    img=plt.imread('C:/Users/Vikas/Downloads/002.png')
  File "C:\Users\Vikas\AppData\Local\Programs\Python\Python37-32\lib\site-packages\matplotlib\pyplot.py", line 2129, in imread
    return matplotlib.image.imread(fname, format)
  File "C:\Users\Vikas\AppData\Local\Programs\Python\Python37-32\lib\site-packages\matplotlib\image.py", line 1426, in imread
    with open(fname, 'rb') as fd:
FileNotFoundError: [Errno 2] No such file or directory: 'C:/Users/Vikas/Downloads/002.png'
>>> img=plt.imread('C:/Users/Vikas/Downloads/b31c4ced285e73f40c884d009c5a7da6.png')
>>> img.shape
(720, 1280, 3)
>>> 
