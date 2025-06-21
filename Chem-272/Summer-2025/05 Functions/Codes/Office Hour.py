# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 18:04:20 2025

@author: MMH_user
"""

S1 = 'packaged by Anaconda' 
S2 = 'IPython 8.20.0 -- An enhanced Interactive Python'
S3 = 'Type "copyright", "credits" or "license" for more information.'
S4 = '| (main, Dec 15 2023, 18:05:47) [MSC v.1916 64 bit (AMD64)]'


#1)
Find_Letter1 = lambda Str: [print(s) for s in Str]

#2)
Find_Letter2 = lambda char, Str: [print(s) for s in Str if s == char]

#3)
Find_Letter3 = lambda char, Str: len([print(s) for s in Str if s == char])

#4)
Find_Letter4 = lambda char, Str: len([s for s in Str if s == char])


S     = [S1, S2, S3, S4]
Chars = ['a','n','i','2']

All_Str = list(map(Find_Letter4, Chars, S))



def Fridge(**My_Stuff):
    
    if My_Stuff:
    
        Optimal = {'beer': 5, 'eggs': 12, 'sausage': 4, 'butter': 1, 'oranges': 5}
        
        Keys = My_Stuff.keys()
        Vals = My_Stuff.values()
        
        
        for k,v in zip(Keys, Vals):
            
            opt_v = Optimal[k]
            
            if opt_v > v:
                print('We only have ' + str(v) + ' ' + k + ' in the fridge.\n ')
                print('We need to buy ' + str(opt_v-v) + ' more ' + k)
                
            else:
                print('We have enough ' + k)
                
    else:
        print('Provide items')
        
        


