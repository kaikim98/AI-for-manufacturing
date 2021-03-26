import os
import pandas as pd
import numpy as np

data = pd.read_csv('big-mac-source-data_modified.csv', index_col=0)
#data_korea = data[data['iso_a3'] == 'KOR']

def bigmac_index(w, b, x, y, ):
    y_hat = w * x + b
    E = 0
    F = 0
    n = len(x)
    for i in range(0, n):
        E = E + x*(y-y_hat)
    w = w - step / n * E
    for i in range(0, n):
        F = F + (y-y_hat)
    b = b - step / n * F
    return w, b

x = np.array(data['local_price'] / data['dollar_ex'])
GDP = np.array(data['GDP_dollar'])
y = np.