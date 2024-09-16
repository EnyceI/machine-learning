# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 14:48:19 2024

@author: ienyc
"""

import pandas as pd
import numpy as np
df=pd.read_csv(r'C:\Users\ienyc\OneDrive\Desktop\ML Project\bloodpressure-23.csv')
x= df[['WEIGHT']].values 
y=df['SYSTOLIC'].values
print(df.columns)
# List to store results
deg=list(range(1,15))
mselist=[]
r2list=[]
rmselist=[] #to store rmses
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import cross_val_score
print("POLYNOMIAL REGRESSION")
print("======================================")

for degree in deg:
    # create polynomial feature
    poly= PolynomialFeatures(degree=degree)
    xpoly=poly.fit_transform(x)
    
    # fit the model
    model=LinearRegression()
    model.fit(xpoly, y)
    
    # predict using the model
    pred=model.predict(xpoly)
    
    # evaluate the model
    mse=mean_squared_error(y,pred)
    r2=r2_score(y,pred)
    
    # 10-fold cross validation
    msescore= -cross_val_score(model, xpoly,y, cv=10, scoring='neg_mean_squared_error')
    r2score=cross_val_score(model,xpoly,y,cv=10,scoring='r2')
    
    # mean of cross validation score
    msemean=msescore.mean()
    r2mean=r2score.mean()
    
    # Calculation of RMSE
    rmsemean=np.sqrt(msemean)
    rmselist.append(rmsemean)
    mselist.append(mse)
    r2list.append(r2)
    #print results
    print(f' Mean MSE: {msemean}')
    print(f'Mean R^2: {r2mean}')
    print(f"Degree:{deg}, MSE:{mse}, R^2:{r2}, Mean RMSE:{rmsemean}")
#best degree
bestdegree=6
bestpolydegree= PolynomialFeatures(degree=bestdegree)
xpolybest=bestpolydegree.fit_transform(x)
bestmodel=LinearRegression()
bestmodel.fit(xpolybest, y)
print("------------------------")
print(" BEST DEGREE:",bestdegree)
print("Intercept:",bestmodel.intercept_)
print("Coeffients:",bestmodel.coef_)
print("---------------------------")
#plot the result
import matplotlib.pyplot as plt

plt.figure(figsize=(16,5))

plt.subplot(1,2,1)
plt.plot(deg,mselist,marker='*')
plt.title('MS vs POLYNOMIAL DEGREE')
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error')

plt.subplot(1,2,2)
plt.plot(deg,r2list,marker='o', color='r')
plt.title('R^2 vs POLYNOMIAL DEGREE')
plt.xlabel('Polynomial Degree')
plt.ylabel('R^2')

#plot for RMSE(error plot)
plt.figure(figsize=(10,6))
plt.plot(deg,rmselist,marker='o',color='r')
plt.title('Mean RMSE VS Polynomial Degree(ERROR PLOT')
plt.xlabel('Ploynomial Degree')
plt.ylabel('Mean RMSE')
plt.grid(True)
plt.show()
# ********MULTIPLE LINEAR REGRESSION***********
non_numerical_column=['GENDER','NAME','MARITAL-STATUS']
df_numeric=df.drop(columns= non_numerical_column)
useful=df_numeric.drop(['SYSTOLIC'],axis=1)
x_mult=useful.values
y_mult=df['SYSTOLIC'].values
mult_model=LinearRegression()
mult_model.fit(x_mult,y_mult)
# 10-FOLD CROSS VALIDATION FOR MULTIPLE LINEAR REGRESSION
mse_mult_score=-cross_val_score(mult_model,x_mult,y_mult,cv=10,scoring='neg_mean_squared_error')
r2_mult_score=cross_val_score(mult_model,x_mult,y_mult,cv=10,scoring='r2')
mse_mult_mean=mse_mult_score.mean()
r2_mult_mean=r2_mult_score.mean()
rmse_mult_mean=np.sqrt(mse_mult_mean)
print("=======================")
print("MULTPILE LINEAR REGRESSION MODEL")
print("Intercept:", mult_model.intercept_)
print("Coefficient: ",mult_model.coef_)
print("Mean MSE:",mse_mult_mean)
print("Mean R2: ",r2_mult_mean)
#print("Mean RMSE:")
y_pred_mult=mult_model.predict(x_mult)
mse_mult=mean_squared_error(y_mult,y_pred_mult)
r2_mult=r2_score(y_mult,y_pred_mult)
print("=================================")
print("Multiple Linear Regression Model MSE:",mse_mult)
print("Multiple Linear Regression Model R2: ",r2_mult)

#***********RIDGE REGRESSION*************
from sklearn.linear_model import Ridge
alpha=0.1
ridge_model=Ridge(alpha=alpha)
ridge_model.fit(x_mult, y_mult)
y_pred_ridge=ridge_model.predict(x_mult)
#10- fold cross validation
mse_ridge_score=-cross_val_score(ridge_model,x_mult,y_mult,cv=10,scoring='neg_mean_squared_error')
r2_ridge_score=cross_val_score(ridge_model,x_mult,y_mult,cv=10,scoring='r2')
#calculation of Mean MSE and R2 
mse_ridge_mean=mse_ridge_score.mean()
r2_ridge_mean=r2_ridge_score.mean()
rmse_ridge_mean=np.sqrt(mse_ridge_mean)
print("=================================")
print("RIDGE REGRESSION MODEL")
print("Intercept:",ridge_model.intercept_)
print("Coefficient: ",ridge_model.coef_)
print("Mean RMSE: ", rmse_ridge_mean)
