#!/usr/bin/env python
# coding: utf-8

# In[36]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import PolynomialFeatures

veri = pd.read_excel("Pythonveri.xls")
veri.head(10)

Minutes = veri[['Minutes']]
Customer = veri[['Customer']]

x_train, x_test,y_train,y_test = train_test_split(Minutes,Customer,test_size=0.33, random_state=0)

sc = StandardScaler()
lr = LinearRegression()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

lr.fit(x_train,y_train)

test = lr.predict(x_test)

x_train = x_train.sort_index()
y_train = y_train.sort_index()


plt.plot(x_train,y_train)
plt.plot(x_test,lr.predict(x_test))


def linearRegressionVisual():
    plt.scatter(veri["Minutes"],
                veri["Customer"],
                s=100,
                c="red",
                edgecolors='red'
                )
    plt.plot(Minutes,lr.predict(Minutes), color='blue')
    plt.title('Doğrusal regresyona göre olan sonuç')
    plt.xlabel('Dakika')
    plt.ylabel('Müşteri')
    plt.grid(True)
    plt.show()
    return
linearRegressionVisual()

poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(Minutes)
pol_reg = LinearRegression()
pol_reg.fit(X_poly, Customer)
def polynomialRegressionVisual():
    plt.scatter(veri["Minutes"],
                veri["Customer"],
                s=100,
                c="red",
                edgecolors='red'
                )
    plt.plot(Minutes, pol_reg.predict(poly_reg.fit_transform(Minutes)), color='blue')
    plt.title('Polinomsal Regresyon Sonucu')
    plt.xlabel('Dakika')
    plt.ylabel('Müşteri')
    plt.grid(True)
    plt.show()
    return

polynomialRegressionVisual()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




