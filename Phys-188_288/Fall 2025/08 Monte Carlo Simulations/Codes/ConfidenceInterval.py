# -*- coding: utf-8 -*-
"""
Created on Thu May 15 14:12:13 2025

@author: MMH_user
"""
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from datetime import datetime
from sklearn.utils import resample
from scipy.optimize import curve_fit

"""
USAGE:
    generating a test sample:
    
    x         = np.linspace(-1,3,20)
    err       = np.random.normal(0, 1, (len(x),))#1sigma errorbars
    y         = x**2 + err
    errorbars = abs(err)
    
    
    1) plotting data
        
    F1 = FitData(x, y)
    F2 = FitData(x, y, errorbars)
    F3 = FitData(x, y, errorbars, time = '[s]', pressure = '[MPa]')
    
    
    2) fitting data (returns best values of fitted params, 1sigma confidence and
                     reduced chi2 if errorbars given, MSE else)
                     
    res1  = F1.Fit()
    res2  = F2.Fit()
    res2  = F3.Fit()
    
    res12 = F1.Fit("a*x**2", [1], (-0.5, 10))
    
    
    3) Bootstrapping (either varying within errorbars or within conf of fitted
                      params)
                      
    F1.RunBootStrap()
    F2.RunBootStrap()
    F3.RunBootStrap()
    
    F1.RunBootStrap(100, [90, 95], np.linspace(-1,5,200))
    F3.RunBootStrap(100, [90, 95], np.linspace(-1,5,200))
"""



#turns string dynamically into a  mathematical function
def StringToFunction(func_str):
   
    common_functions = {'sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs',
                        'arcsin', 'arccos', 'arctan', 'sinh', 'cosh', 'tanh', 'pi'}

    all_vars = set(re.findall(r'\b(?!x\b|np\b)\b[a-zA-Z_][a-zA-Z0-9_]*\b', func_str))
    params = [var for var in all_vars if var not in common_functions]

    def dynamic_func(x, *args):
        local_vars = {param: args[i] for i, param in enumerate(params)}
        local_vars['x'] = x
        local_vars['np'] = np
        return eval(func_str, {}, local_vars)

    return dynamic_func, params

###############################################################################

###############################################################################
#Creates plot of the confidence band
def PlotConfIntervall(x_pred, y_predicted, Conf):
    
    for v in Conf:
        alpha      = v/100
        y_ci_lower = np.percentile(y_predicted, (1 - alpha) / 2 * 100, axis = 0)
        y_ci_upper = np.percentile(y_predicted, (1 + alpha) / 2 * 100, axis = 0)
        
        plt.fill_between(x_pred, y_ci_lower, y_ci_upper, color = 'r', alpha = 0.1,\
                     label = str(v) + '% Confidence Interval')
    plt.legend()
###############################################################################

###############################################################################
def PlotData(X: np.array, Y: np.array, *errorbars: np.array, **units: dict):
    
    if errorbars:
        plt.errorbar(X, Y, yerr = errorbars, fmt = '.', c = 'k')

    else:
        plt.scatter(X, Y, marker = '.', c = 'k')
    plt.title("data")
    
    if units:
        quantity = list(units.keys())
        unit     = list(units.values())
        plt.xlabel(quantity[0] + " " + unit[0])
        plt.ylabel(quantity[1] + " " + unit[1])

###############################################################################

###############################################################################
def PlotFit(X_pred, Y_pred):
    plt.plot(X_pred, Y_pred, '-', c = 'r')
###############################################################################

###############################################################################
#Performs bootstrapping for calculating conf intervals/bands
def BootStrap(N_Boot, X, Y, X_pred, ValsBest, Param1Sigma, Fit_Function, Start, Bounds, *errorbars):
    
    y_predicted = np.zeros((N_Boot, len(X_pred)))
        
    if not errorbars:#varying conf intervals of fit params
        for i in range(N_Boot):
            ValsBest_boot     = np.random.normal(ValsBest, Param1Sigma)
            y_predicted[i, :] = Fit_Function(X_pred, *ValsBest_boot)
        
    else:#varying within errorbars
        errorbars = errorbars[0]#tuple to array
        for i in range(N_Boot):
            y_boot            = np.random.normal(Y, errorbars)
            ValsBest_boot, _  = curve_fit(Fit_Function, X, y_boot, p0 = Start,\
                                          bounds = Bounds)
            y_predicted[i, :] = Fit_Function(X_pred, *ValsBest_boot)
            
    return y_predicted
    
###############################################################################

###############################################################################

class FitData():
    
    def __init__(self, X: np.array, Y: np.array, *errorbars: np.array, **units: dict):
        #provide units if needed: time = "[s]", pressure = "[MPa]"
        
        #plotting--------------------------------------------------------------
        PlotData(X, Y, *errorbars, **units)
        
        if errorbars:
            self.E = errorbars
        if units:
            self.U = units
            
        self.X = X
        self.Y = Y
        #----------------------------------------------------------------------
        
    def Fit(self, Function_string: str = "a*x**2 + b*x + c",
                  Start = [1,1,1], Bounds: tuple = (0, [3., 2., 1.])):
        
        Fit_Function, _ = StringToFunction(Function_string)
            
        X       = self.X
        Y       = self.Y

        X_pred  = np.linspace(min(X), max(X), 300)
        
    
        if hasattr(self, "E"):
            #error weighted fit
            ValsBest, Cov = curve_fit(Fit_Function, X, Y, p0 = Start,\
                                      bounds = Bounds, sigma = self.E[0],\
                                      absolute_sigma = True)
                
            Y_fit         = Fit_Function(X, *ValsBest)
            Y_pred        = Fit_Function(X_pred, *ValsBest)
            #red chi2 
            diff   = ((Y_fit - Y)**2) / self.E[0]**2
            Metric = np.sum(diff)/(len(X) - len(Start) -1)
            
            #plotting data
            if hasattr(self, "U"):
                PlotData(X, Y, *self.E, **self.U)
            else:
                PlotData(X, Y, *self.E)
            
        else:
            ValsBest, Cov = curve_fit(Fit_Function, X, Y, p0 = Start,\
                                      bounds = Bounds)
                
            Y_fit         = Fit_Function(X, *ValsBest)
            Y_pred        = Fit_Function(X_pred, *ValsBest)
            
            #mse
            Metric = np.mean((Y_fit - Y)**2)
            
            #plotting data
            if hasattr(self, "U"):
                PlotData(X, Y, **self.U)
            else:
                PlotData(X, Y)
            
        #plot fit
        PlotFit(X_pred, Y_pred)
        
        Result = [ValsBest, np.sqrt(np.diagonal(Cov)), Metric]
        
        
        self.Fit_Function = Fit_Function
        self.Result       = Result
        self.Start        = Start
        self.Bounds       = Bounds
    
        return Result
    
    def RunBootStrap(self, N_Boot: int = 50, Conf: list = [68, 90, 95, 99],
                           *X_predict):
        if not X_predict:
            X_predict = self.X
        else:
            X_predict = X_predict[0]#tuple to array
            
        
        #actual bootstrapping
        if hasattr(self, "E"):
            y_predicted = BootStrap(N_Boot, self.X, self.Y, X_predict,\
                                    self.Result[0], self.Result[1],\
                                    self.Fit_Function, self.Start,\
                                    self.Bounds, *self.E)
                
            PlotConfIntervall(X_predict, y_predicted, Conf)
            
            if hasattr(self, "U"):
                PlotData(self.X, self.Y, *self.E, **self.U)
            else:
                PlotData(self.X, self.Y, *self.E)
                
        else:
            y_predicted = BootStrap(N_Boot, self.X, self.Y, X_predict,\
                                    self.Result[0], self.Result[1],\
                                    self.Fit_Function, self.Start,\
                                    self.Bounds)
            
            PlotConfIntervall(X_predict, y_predicted, Conf)
                
            if hasattr(self, "U"):
                PlotData(self.X, self.Y, **self.U)
            else:
                PlotData(self.X, self.Y)
        
        Y_predict = self.Fit_Function(X_predict, *self.Result[0])
        PlotFit(X_predict, Y_predict)
    
    