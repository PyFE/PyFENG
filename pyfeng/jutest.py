# -*- coding: utf-8 -*-
"""
Created on Sat May  1 20:17:09 2021

@author: Lantian
"""

class test():
    def __init__ (self, a, b):
        self.a = a
        self.b = b
        
    def minus(self):
        return self.a -self.b
    
    def plus(self):
        return self.minus()+2*self.b

a = test(1,2)
print(a.plus())