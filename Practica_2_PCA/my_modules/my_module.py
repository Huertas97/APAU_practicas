#Example of definition of a new module.
#A module is a file containing Python definitions and statements.
#The file name is the module name with the suffix .py appended.
#Within a module, the module’s name (as a string) is available as
#the value of the global variable __name__.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)

#You can define new functions

##def my_function(variables):
##    Your code
##    return variables
