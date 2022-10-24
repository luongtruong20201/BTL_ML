from ast import main
from tkinter import *
from tkinter import ttk
from sklearn import metrics, linear_model, model_selection, decomposition
import numpy as np
import pandas as pd


def __pca(n, data_X):
    pca = decomposition.PCA(n_components = n)
    pca.fit(data_X)
    
    return pca


df = pd.read_excel(
    io='./Folds5x2_pp.xlsx',
    sheet_name=None
)

data = pd.concat(df).to_numpy()


X = data[:, :-1]
Y = data[:, -1:]

form = Tk()
form.title("Machine Learning")
form.geometry('300x200')

mainframe = ttk.Frame(form, padding="3 3 25 25")
mainframe.grid(column=0, row=0, sticky=(N,W,E,S))
form.columnconfigure(0, weight=1)
form.rowconfigure(0, weight=1)

kfold_training_set = StringVar()
kfold_training_set = ttk.Entry(mainframe, width=15, textvariable=kfold_training_set)
kfold_training_set.grid(column=2, row=1, sticky=(W, E))

pca_input = StringVar()
pca_input_Entry = ttk.Entry(mainframe, width=15, textvariable=pca_input)
pca_input_Entry.grid(column=2, row=2, sticky=(W,E))

labelW0 = ttk.Label(mainframe)
labelW = ttk.Label(mainframe)
labelScore = ttk.Label(mainframe)

reg = ''
w0 = ''
w = ''
score = ''
pca = ''
entry=''
lb = ''

labelW0.grid(column=1, row=5, sticky=E)
labelW.grid(column=1, row=6, sticky=E)
labelScore.grid(column=1, row=7, sticky=E)

def calculate():
    form.geometry("400x250")
    global X, Y, w0, w, score, pca, reg
    pca_num = int(pca_input.get())
    pca = __pca(pca_num, X)
    
    pca.fit(X)
    x_new = pca.transform(X)

    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        x_new, Y,
        shuffle=False,
        random_state=None,
        test_size=0.3,
    )
    
    n_kfold = int(kfold_training_set.get())
    kfold = model_selection.KFold(
        shuffle=False,
        random_state=None,
        n_splits=n_kfold
    )
    
    
    Min = 1e10000000
    for(train_index, val_index) in kfold.split(x_train):
        X_train, Y_train = x_train[train_index], y_train[train_index]
        X_val, Y_val = x_train[val_index], y_train[val_index]
        
        linear = linear_model.LinearRegression()
        
        linear.fit(X_train, Y_train)
        
        Y_train_pred = linear.predict(X_train)
        Y_val_pred = linear.predict(X_val)
        
        sum_error = metrics.mean_squared_error(
            Y_train_pred,
            Y_train
        ) + metrics.mean_squared_error(
            Y_val_pred,
            Y_val
        )
        
        
        if(Min > sum_error):
            min = sum_error
            reg = linear
            
    w0 = reg.intercept_
    w = reg.coef_
    score = reg.score(x_test, y_test)
    labelW0.config(text='W[0]: {0}'.format(w0))
    labelW.config(text='W: {0}'.format(w))
    labelScore.config(text='Coefficient of determination: {0}'.format(score))
    

def getResult():
    global entry, reg, lb, pca
    x =[] 
    for i in range(len(entry)):
        x.append(float(entry[i].get()))
    
    x = np.array([x])
    x_new = pca.transform(x)
    y_pred = reg.predict(x_new)
    lb.config(text='Predicted Result: {0}'.format(y_pred))

columnName = ["AT","V","AP","RH"]
def predictResult():
    global reg, entry, lb
    pca_num = int(pca_input.get())
    newWindow = Toplevel(mainframe)
    newWindow.title('Predict the result: ')
    newWindow.geometry('400x200')
    entry = [StringVar() for i in range(4)]
    for i in range(4):
        ttk.Label(newWindow, text=columnName[i]).grid(row=i, column=1)
        Entry(newWindow, textvariable=entry[i]).grid(row=i, column=2)
    for child in newWindow.winfo_children():
        child.grid_configure(padx=5, pady=5)
    lb = Label(newWindow, text='')
    lb.grid(row=5, column=3)
    Button(newWindow, text='Predict', command=getResult).grid(row=4, column=2)

            
ttk.Button(mainframe, text="Calculate", command=calculate).grid(column=2, row=3, sticky=W)
ttk.Label(mainframe, text="Kfold training set: ").grid(column=1, row=1, sticky=E)
ttk.Label(mainframe, text="PCA dimension: ").grid(column=1, row=2, sticky=E)

ttk.Button(mainframe, text="Predict", command=predictResult).grid(column=2, row=4, sticky=W)

form.mainloop()