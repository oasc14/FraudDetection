# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 00:42:32 2022

@author: Oscar A Suarez Castillo

the ord() function returns the unicode from a given character, example:
    ord(c)=99
for this database i used this transform in the Sucursal, Territorio and
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





