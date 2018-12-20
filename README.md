# RBFinterpolate
Interpolate a 1-D function by radial basis function.
'x' and 'y' are arrays of value used to approximate some function f:'y = f(x)'.
This class returns a function whose call method uses interpolation to fin the value of new points.

## Parameter
x : ndarray  
&nbsp; &nbsp; &nbsp; &nbsp; 
A 1-D array of real values.  

y : ndarray  
&nbsp; &nbsp; &nbsp; &nbsp; 
A 1-D array of real values. The length of 'y' must be equal to the length of 'x'.

mode : str, optional  
&nbsp; &nbsp; &nbsp; &nbsp; 
Specify whether to perform regularization as a string ('interpolate' or 'fitting').  
&nbsp; &nbsp; &nbsp; &nbsp; 
Default is 'interpolate'.

kind : str, optional  
&nbsp; &nbsp; &nbsp; &nbsp; 
Specifies the kind of interpolation as a string ('linear', 'gaussian', 'multiquadric',  
&nbsp; &nbsp; &nbsp; &nbsp; 
'inv-quadric', 'inv-multiquadric', 'thin-plate').  
&nbsp; &nbsp; &nbsp; &nbsp; 
Default is 'gaussian'.

eps : int or float, optional  
&nbsp; &nbsp; &nbsp; &nbsp; 
A parameter that determines the intensity of the radial basis function.  
&nbsp; &nbsp; &nbsp; &nbsp; 
It is effective only when kind is (gaussian', 'multiquadric', 'inv-quadric', 'inv-multiquadric').  
&nbsp; &nbsp; &nbsp; &nbsp; 
Default is 1.0.

lamda : int or float, optional  
&nbsp; &nbsp; &nbsp; &nbsp; 
A parameter that determines smoothness of interpolation function.  
&nbsp; &nbsp; &nbsp; &nbsp; 
It is effective only when mode is 'fitting'.  
&nbsp; &nbsp; &nbsp; &nbsp; 
lamda must be larger than 0.0.  
&nbsp; &nbsp; &nbsp; &nbsp; 
Default is 0.0.

## Methods

* __radial_basis_function  
* __weight_solve  
* __interpolate  
* \_\_call__

## Example
``` py3
import numpy as np
from RBFinterpolate.interpolate import Interp1dRBF
import matplotlib.pyplot as plt
x = np.arange(0.0, 2.0 * np.pi, 0.25)
y = np.sin(x)
f = Interp1dRBF(x, y, kind='multiquadric', eps=2.0)    
xnew = np.arange(0.0, 2.0 * np.pi, 0.01)
ynew = f(xnew)
plt.plot(x, y, 'o', xnew, ynew, '-')
plt.show()
```

