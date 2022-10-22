from random import random
import numpy as np
import pandas as pd
from sklearn import linear_model, metrics, model_selection, decomposition

df = pd.read_excel(
    io='./Folds5x2_pp.xlsx',
    sheet_name = None
)

df = pd.concat(df)

data = df.to_numpy()

data_X = data[:, :-1]
data_y = data[:, -1:]

result = []
MAX = -1e100000

_pca = []

for i in range(1, data_X.shape[1]):
    min = 1e10000
    print("Lan thu %s:" %i)
    pca = decomposition.PCA(n_components = i)
    print(pca)
    pca.fit(data_X)
    
    x = pca.transform(data_X)
    reg = ''
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
        x, data_y,
        test_size = 0.3,
        shuffle=False,
        random_state = None
    )
    
    kfold = model_selection.KFold(
        shuffle=False,
        random_state=None,
        n_splits=5
    )
    

    for train_index, valid_index in kfold.split(X_train):
        x_train = X_train[train_index]; x_valid = X_train[valid_index]
        y_train = Y_train[train_index]; y_valid = Y_train[valid_index]
        
        linear = linear_model.LinearRegression()
        
        linear.fit(
            x_train,
            y_train
        )
        
        y_train_pred = linear.predict(x_train)
        y_valid_pred = linear.predict(x_valid)
        
        sum_error = metrics.mean_squared_error(
            y_train_pred,
            y_train
        ) + metrics.mean_squared_error(
            y_valid_pred,
            y_valid
        )
        
        if(sum_error < min):
            min = sum_error
            reg = linear
    result.append(reg)
    score = reg.score(X_test, Y_test)
    
    if(score > MAX):
        MAX = score
        _pca = pca
        
print(MAX)
print(_pca)

inp = [float(input("Nhap he so thu {0}: ".format(i))) for i in range(4)]

inp = _pca.transform(np.array([inp]))
print(result[-1].predict(inp))