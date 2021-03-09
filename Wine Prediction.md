```python
# Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split   
from sklearn import preprocessing    
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline 
from sklearn.model_selection import GridSearchCV    
from sklearn.metrics import mean_squared_error, r2_score  
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
# Load red wine data from remote url
dataset_url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv")
data = pd.read_csv(dataset_url)
```


```python
# Output the first 5 rows of data
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed acidity;"volatile acidity";"citric acid";"residual sugar";"chlorides";"free sulfur dioxide";"total sulfur dioxide";"density";"pH";"sulphates";"alcohol";"quality"</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.4;0.7;0;1.9;0.076;11;34;0.9978;3.51;0.56;9.4;5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.8;0.88;0;2.6;0.098;25;67;0.9968;3.2;0.68;9.8;5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.8;0.76;0.04;2.3;0.092;15;54;0.997;3.26;0.65;...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11.2;0.28;0.56;1.9;0.075;17;60;0.998;3.16;0.58...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7.4;0.7;0;1.9;0.076;11;34;0.9978;3.51;0.56;9.4;5</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Read CSV with semicolon separator
data = pd.read_csv(dataset_url, sep=';')
 
data.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.4</td>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.9978</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.8</td>
      <td>0.88</td>
      <td>0.00</td>
      <td>2.6</td>
      <td>0.098</td>
      <td>25.0</td>
      <td>67.0</td>
      <td>0.9968</td>
      <td>3.20</td>
      <td>0.68</td>
      <td>9.8</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.8</td>
      <td>0.76</td>
      <td>0.04</td>
      <td>2.3</td>
      <td>0.092</td>
      <td>15.0</td>
      <td>54.0</td>
      <td>0.9970</td>
      <td>3.26</td>
      <td>0.65</td>
      <td>9.8</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11.2</td>
      <td>0.28</td>
      <td>0.56</td>
      <td>1.9</td>
      <td>0.075</td>
      <td>17.0</td>
      <td>60.0</td>
      <td>0.9980</td>
      <td>3.16</td>
      <td>0.58</td>
      <td>9.8</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7.4</td>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.9978</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
print (data.shape)
```

    (1599, 12)



```python
data.describe
```




    <bound method NDFrame.describe of       fixed acidity  volatile acidity  citric acid  residual sugar  chlorides  \
    0               7.4             0.700         0.00             1.9      0.076   
    1               7.8             0.880         0.00             2.6      0.098   
    2               7.8             0.760         0.04             2.3      0.092   
    3              11.2             0.280         0.56             1.9      0.075   
    4               7.4             0.700         0.00             1.9      0.076   
    ...             ...               ...          ...             ...        ...   
    1594            6.2             0.600         0.08             2.0      0.090   
    1595            5.9             0.550         0.10             2.2      0.062   
    1596            6.3             0.510         0.13             2.3      0.076   
    1597            5.9             0.645         0.12             2.0      0.075   
    1598            6.0             0.310         0.47             3.6      0.067   
    
          free sulfur dioxide  total sulfur dioxide  density    pH  sulphates  \
    0                    11.0                  34.0  0.99780  3.51       0.56   
    1                    25.0                  67.0  0.99680  3.20       0.68   
    2                    15.0                  54.0  0.99700  3.26       0.65   
    3                    17.0                  60.0  0.99800  3.16       0.58   
    4                    11.0                  34.0  0.99780  3.51       0.56   
    ...                   ...                   ...      ...   ...        ...   
    1594                 32.0                  44.0  0.99490  3.45       0.58   
    1595                 39.0                  51.0  0.99512  3.52       0.76   
    1596                 29.0                  40.0  0.99574  3.42       0.75   
    1597                 32.0                  44.0  0.99547  3.57       0.71   
    1598                 18.0                  42.0  0.99549  3.39       0.66   
    
          alcohol  quality  
    0         9.4        5  
    1         9.8        5  
    2         9.8        5  
    3         9.8        6  
    4         9.4        5  
    ...       ...      ...  
    1594     10.5        5  
    1595     11.2        6  
    1596     11.0        6  
    1597     10.2        5  
    1598     11.0        6  
    
    [1599 rows x 12 columns]>




```python
# Separate target from training features
y = data.quality    
X = data.drop('quality', axis=1)
```


```python
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=123, 
                                                    stratify=y)
```


```python
# Declare datapreprocessing steps
```


```python
# Fitting the transformer API
scaler = preprocessing.StandardScaler().fit(X_train) 
```


```python
# Applying transformer to training data
X_train_scaled = scaler.transform(X_train)
print( X_train_scaled.mean(axis=0))
print( X_train_scaled.std(axis=0))
```

    [ 1.16664562e-16 -3.05550043e-17 -8.47206937e-17 -2.22218213e-17
      2.22218213e-17 -6.38877362e-17 -4.16659149e-18 -2.54439854e-15
     -8.70817622e-16 -4.08325966e-16 -1.17220107e-15]
    [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]



```python
# Applying transformer to testing data
X_test_scaled = scaler.transform(X_test)
print( X_test_scaled.mean(axis=0))
print( X_test_scaled.std(axis=0))
```

    [ 0.02776704  0.02592492 -0.03078587 -0.03137977 -0.00471876 -0.04413827
     -0.02414174 -0.00293273 -0.00467444 -0.10894663  0.01043391]
    [1.02160495 1.00135689 0.97456598 0.91099054 0.86716698 0.94193125
     1.03673213 1.03145119 0.95734849 0.83829505 1.0286218 ]



```python
# Pipeline with preprocessing and model
pipeline = make_pipeline(preprocessing.StandardScaler(), 
                         RandomForestRegressor(n_estimators=100))
```


```python
# List tunable hyperparameters
print (pipeline.get_params)


```

    <bound method Pipeline.get_params of Pipeline(steps=[('standardscaler', StandardScaler()),
                    ('randomforestregressor', RandomForestRegressor())])>



```python
# Declare hyperparameters to tune
hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                  'randomforestregressor__max_depth': [None, 5, 3, 1]}
```


```python
clf = GridSearchCV(pipeline, hyperparameters, cv=10)

```


```python
# Fit and tune model
clf.fit(X_train, y_train)
```




    GridSearchCV(cv=10,
                 estimator=Pipeline(steps=[('standardscaler', StandardScaler()),
                                           ('randomforestregressor',
                                            RandomForestRegressor())]),
                 param_grid={'randomforestregressor__max_depth': [None, 5, 3, 1],
                             'randomforestregressor__max_features': ['auto', 'sqrt',
                                                                     'log2']})




```python
print (clf.best_params_)

```

    {'randomforestregressor__max_depth': None, 'randomforestregressor__max_features': 'log2'}



```python
# Confirm model will be retrained
print (clf.refit)

```

    True



```python
plt.figure
```


```python
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True)

```




    <AxesSubplot:>




    
![png](output_20_1.png)
    



```python
# Strong correlation between
# fixed acidity and citric acid, fixed acidity and density, free sulfur dioxide and total sulfur dioxide,
# alcohol and quality
# Both variables will increase
```


```python
# Weak correlation between
# volatile acidity and citric acid, fixed acidity and ph, ph and citric acid
# One variable will increase while the other variable decreases
```
