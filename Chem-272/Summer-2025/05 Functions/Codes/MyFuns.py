# -*- coding: utf-8 -*-
"""
Created on Tue May 13 21:42:35 2025

@author: MMH_user
"""
#2 positional input arguments
def MyFun1(a: float, b: float):
    res = a*b
    return res

#one argument with a default setting
def MyFun2(a: float, b: float = 4):
    res = a*b
    return res

#variable number of input arguments
def MyFun3(*my_args):
    
    Sentence = ''
    
    for a in my_args:
        Sentence += ' ' + a
    
    return Sentence[1:]
    
#variable number of keyword input arguments
def MyFun4(**my_args):
    print(my_args)
    
    
    
def MyFun5(a: float, b: float = 4, *words, **groceries):
    
    res1 = a+b
    res2 = a*b
    res3 = a**b
    
    if words:
        Sentence = ''
        
        for w in words:
            Sentence += ' ' + w
        print(Sentence[1:])
    
    if groceries:
        print(groceries)
    
    return res1, res2, res3

