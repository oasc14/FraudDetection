# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 00:42:32 2022

@author: Oscar A Suarez Castillo

----------------------------------------------
This script use a Self Organized Map for a fraud detection
----------------------------------------------

*The ord() function returns the unicode from a given character, example: 
    ord(c)=99

*For this database i used this transform in the Sucursal, Territorio and
Cliente for made a change in the values of the database from string to
float. This change facilite the calculus.

"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Import 
dataset = pd.read_csv('./DatasetPagos.csv')

# -------------------Data Wrangling---------------------

# Cleaning the dataset

# Quit NaN values 

dataset['Contrato_con_Fraude']=dataset['Contrato_con_Fraude'].fillna(0)


# droping the last raw 
dataset = dataset.drop(dataset.index[30572])

#dataset['Contrato_con_Fraude']=dataset['Contrato_con_Fraude'].replace('nan',0)

# Changing Sucursal ord(s) = 83
dataset['Sucursal'] = dataset['Sucursal'].str.replace('Sucursal','83')

# Changing Territorio ord(M)=77, ord(E)=69, ord(X)=120
dataset['Territorio'] = dataset['Territorio'].str.replace('MEX-','7769120')

# Changing Sucursal ord(C)=99
dataset['Cliente'] = dataset['Cliente'].str.replace('C','99')

dataset['Agreement_Num'] = dataset['Agreement_Num'].str.replace('AG','6571')

# Changing Fecha_Pago to datetime
dataset['Fecha_Pago']=pd.to_datetime(dataset['Fecha_Pago'])

# Quit white spaces in the database
def remove_whitespace(x):
    try:
        x = ''.join(x.split())
    except:
        pass
    return x
dataset.Sucursal = dataset.Sucursal.apply(remove_whitespace)

# Changing values to float
dataset['Sucursal'] = pd.to_numeric(dataset['Sucursal'])
dataset['Territorio'] = pd.to_numeric(dataset['Territorio'])
dataset['Cliente'] = pd.to_numeric(dataset['Cliente'])
dataset['Agreement_Num'] = pd.to_numeric(dataset['Agreement_Num'])
dataset['Fecha_Pago']=pd.to_numeric(dataset['Fecha_Pago'])
dataset['Contrato_con_Fraude']=dataset['Contrato_con_Fraude'].astype(int)

#-------------------Finish Data Wrangling-----------------------

# Split data
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:,-1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
X = sc.fit_transform(X)

# NAO Train
from minisom import MiniSom 
som = MiniSom(x=10,y=10,input_len=9, sigma=1.0, learning_rate=0.5)
som.random_weights_init(X)
som.train_random(data=X, num_iteration=4000)

# Show the results
from pylab import bone,pcolor,colorbar,plot,show
bone()
pcolor(som.distance_map())
colorbar()
markers=['o','s']
colors=['r','b']
for i,x in enumerate(X):
    w = som.winner(x)
    plot(w[0]+0.5,
         w[1]+0.5,
         markers[Y[i]],
         markeredgecolor = colors[Y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

# detecting fraud
mapeos = som.win_map(X)
fraud = np.concatenate((mapeos[(2,7)],mapeos[(3,3)],mapeos[(8,4)],mapeos[(2,1)]), axis=0)
fraud = sc.inverse_transform(fraud)