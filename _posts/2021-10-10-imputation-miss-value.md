---
layout: post
title: 特征工程中的缺失值插补
description: "以House Price为例快速构建一个Base Line模型"
modified: 2020-10-01
tags: [Dataset, Compete]
categories: [kaggle]
image:
  path: /images/feature_engineering/imputation_values/imputation-value-1.jpg
  feature: /feature_engineering/imputation_values/imputation-value-1.jpg
  credit: imputation-value-1
  creditlink: https://deeplearn.org/arxiv/69226/a-comparative-study-on-hierarchical-navigable-small-world-graphs
---

特征工程中的缺失值补插

因为各种各样的原因，在现实世界中的数据包含各种各样的缺失值，通常以空白、NaN或者其他占位符表示。这样的数据
和scikit-learn的不相兼容，在scikit-learn中假设所有的数据是数值型的数组，并且这些数据都是有具体意义的。在处理
这些不兼容数据的时候，一个最基本的策略就是丢弃对应的行或者对应的列。这种策略的代价是丢失可能丢失掉有价值的数据。
一个更好的策略就是对这些数据进行imputation(补插)以填充数据缺失字段。

# 1. Univariate vs. Multivariate Imputation
在变量的补插当中，一种最基本的补插方法是单变量补插(Univariate Imputation)。补插第i个特征的数据的时候，我们只需要
使用在这个特征中所有没有缺失的数据进行插入。

多变量补插(Multivariate Imputation)和单变量的补插相对，多变量补插算法使用整个数据的变量特征来估计缺失数据。

# 2. Univariate Imputation
在scikit-learn的`SimpleImputer`类中提供了最基本的缺失值补插策略。缺失值可以插入一个提供的常量，或者使用这个特征
所在列的均值、中间值或者最高频率的值。这个类也同样支持对不同的缺失值进行编码。

在接下来的例子中给出了怎么样使用均值来补插特征中的包含的np.nan数据缺失。

```python
>>> import numpy as np
>>> from sklearn.impute import SimpleImputer
>>> imp = SimpleImputer(missing_values=np.nan, strategy='mean')
>>> imp.fit([[1, 2], [np.nan, 3], [7, 6]])
SimpleImputer()
>>> X = [[np.nan, 2], [6, np.nan], [7, 6]]
>>> print(imp.transform(X))
[[4.          2.        ]
 [6.          3.666...]
 [7.          6.        ]]
```

当然这个`SimpleImputer`也支持Sparse矩阵

```python
>>> import scipy.sparse as sp
>>> X = sp.csc_matrix([[1, 2], [0, -1], [8, 4]])
>>> imp = SimpleImputer(missing_values=-1, strategy='mean')
>>> imp.fit(X)
SimpleImputer(missing_values=-1)
>>> X_test = sp.csc_matrix([[-1, 2], [6, -1], [7, 6]])
>>> print(imp.transform(X_test).toarray())
[[3. 2.]
 [6. 3.]
 [7. 6.]]
```

当然`SimplerImputer`也支持用字符串或者pandas类别表示的定类数据。在填充定类的数据的时候包含`most_frequent`策略
和`constant`策略。其中`most_frequent`表示用高频类别填充，`constant`表示用固定类别填充。

```python
>>> import pandas as pd
>>> df = pd.DataFrame([["a", "x"],
...                    [np.nan, "y"],
...                    ["a", np.nan],
...                    ["b", "y"]], dtype="category")
...
>>> imp = SimpleImputer(strategy="most_frequent")
>>> print(imp.fit_transform(df))
[['a' 'x']
 ['a' 'y']
 ['a' 'y']
 ['b' 'y']]
```




# 3. Multivariate feature imputation
当然，还有更为复杂的imputation方式。在数据处理的时候可以使用scikit-learn中的`IterativeImputer`类。
在这个类中，对每一个包含缺失值的数据特征，采用其他特征进行建模并对相应的缺失值进行预估。这个类以循环迭代
的方式进行:在每一次迭代中，处理一个缺失的特征列y，并且将其他特征作为输入X。然后在(X,y)上训练一个回归器
这样就可以通过预测得到y的值。在每一轮迭代中，分别对每个缺失值进行预测，并最终返回补插完后的结果。

```python
>>> import numpy as np
>>> from sklearn.experimental import enable_iterative_imputer
>>> from sklearn.impute import IterativeImputer
>>> imp = IterativeImputer(max_iter=10, random_state=0)
>>> imp.fit([[1, 2], [3, 6], [4, 8], [np.nan, 3], [7, np.nan]])
IterativeImputer(random_state=0)
>>> X_test = [[np.nan, 2], [6, np.nan], [np.nan, 6]]
>>> # the model learns that the second feature is double the first
>>> print(np.round(imp.transform(X_test)))
[[ 1.  2.]
 [ 6. 12.]
 [ 3.  6.]]
```

## 3.1 迭代器的灵活性
在R中有许多构建好的imputation包例如:Amelia, mi, mice, missForest等等。其中missForest是最重要的一个库，
这个库是不同的imputation库的一个重要实现。这些算法都能够通过把回归器传入到`IterativeImputer`中来预测
缺失的特征值。在missForest中，这个回归器是随机森林。

## 3.2 

# 4 Nearest neighbors imputation
在`KNNImputer`类中提供了一个方法用K近邻来填充缺失值。在默认情况下，`KNNImputer`支持欧氏距离来寻找最近邻。
每个确实值的imputation值是其最近的K个邻居在这个特征上对应的值。我们将这些邻居的在这个特征上的取值求平均
或者通过距离进行加权。如果一条数据有多个缺失值，对于不同特征上的进行的imputation可以有不同的邻居节点。
如果整个数据集上不够k个邻居或者是没有定义距离的度量，
那么这个时候选取数据集上这个特征字段的平均值作为imputation值。

关于`KNNImputer`的使用Demo如下:

```python
>>> import numpy as np
>>> from sklearn.impute import KNNImputer
>>> nan = np.nan
>>> X = [[1, 2, nan], [3, 4, 3], [nan, 6, 5], [8, 8, 7]]
>>> imputer = KNNImputer(n_neighbors=2, weights="uniform")
>>> imputer.fit_transform(X)
array([[1. , 2. , 4. ],
       [3. , 4. , 3. ],
       [5.5, 6. , 5. ],
       [8. , 8. , 7. ]])
```

# 5 Marking imputed values

在scikit-learn中`MissingIndicator`可以将数据集中的数据转换为一个二进制矩阵从而指示缺失值的存在。在中转化
和imputation一起使用是非常有效的。在做缺失数据的imputation的时候，
保存哪些数据是丢失的可以提供一些额外的信息。

值得注意的是在`SimplerImputer`和`IterativeImputer`中存在一个bool类型的参数用来指示是否需要添加`MissingIndicator`.

在下面的例子中给出了`MissingIndicator`使用的例子，其中我们用-1表示缺失值:
```python
>>> from sklearn.impute import MissingIndicator
>>> X = np.array([[-1, -1, 1, 3],
...               [4, -1, 0, -1],
...               [8, -1, 1, 0]])
>>> indicator = MissingIndicator(missing_values=-1)
>>> mask_missing_values_only = indicator.fit_transform(X)
>>> mask_missing_values_only
array([[ True,  True, False],
       [False,  True,  True],
       [False,  True, False]])
```


[[1] Imputation of missing values](https://scikit-learn.org/stable/modules/impute.html)