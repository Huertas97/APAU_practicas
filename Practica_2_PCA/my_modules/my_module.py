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



# Función empleada para calcular el numero de componentes necesarios para retener  una
# cantidad deseada de la varianza en una PCA.
# 
# Args: 
# 	-  pca_variance_ratio (list): varianza explicada por cada componente de los datos iniciales
# 	-  var_expl_deseada (float): cantidad de la varianza que se desea retener
# 
# Returns:
# 	- n_componentes (int): numero de componentes necesarios para explicar la varianza deseada en una PCA
# 	- varianza_expl  (float): cantidad de varianza explicada por n_componentes

def num_componentes(pca_variance_ratio, var_expl_deseada = 0.99):
    n_componentes = sum(np.cumsum(pca_variance_ratio) < var_expl_deseada) +1 
    
    if n_componentes == len(pca_variance_ratio):
        varianza_expl = np.cumsum(pca_variance_ratio)[len(pca_variance_ratio)-1]
         
    else:
        varianza_expl = np.cumsum(pca_variance_ratio)[n_componentes-1]
    
    return n_componentes, varianza_expl
