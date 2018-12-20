import sys
import numpy as np

class Interp1dRBF:
    '''
    Interpolate a 1-D function by radial basis function.
    'x' and 'y' are arrays of value used to approximate some function f:'y = f(x)'.
    This class returns a function whose call method uses interpolation to fin the value of new points.

    Parameter
    ----
    x : ndarray
        A 1-D array of real values.
    y : ndarray
        A 1-D array of real values. The length of 'y' must be equal to the length of 'x'.
    mode : str, optional
        Specify whether to perform regularization as a string ('interpolate' or 'fitting').
        Default is 'interpolate'.
    kind : str, optional
        Specifies the kind of interpolation as a string ('linear', 'gaussian', 'multiquadric', 'inv-quadric', 'inv-multiquadric', 'thin-plate').
        Default is 'gaussian'.
    eps : int or float, optional
        A parameter that determines the intensity of the radial basis function.
        It is effective only when kind is (gaussian', 'multiquadric', 'inv-quadric', 'inv-multiquadric').
        Default is 1.0.
    lamda : int or float, optional
        A parameter that determines smoothness of interpolation function.
        It is effective only when mode is 'fitting'.
        lamda must be larger than 0.0.
        Default is 0.0.

    Methods
    ----
    __radial_basis_function
    __weight_solve
    __interpolate
    __call__

    Example
    ----
    >>> import numpy as np
    >>> from RBFinterpolate.interpolate import Interp1dRBF
    >>> import matplotlib.pyplot as plt

    >>> x = np.arange(0.0, 2.0 * np.pi, 0.25)
    >>> y = np.sin(x)
    >>> f = Interp1dRBF(x, y, kind='multiquadric', eps=2.0)

    >>> xnew = np.arange(0.0, 2.0 * np.pi, 0.01)
    >>> ynew = f(xnew)
    >>> plt.plot(x, y, 'o', xnew, ynew, '-')
    >>> plt.show()
    '''

    def __radial_basis_function(self, x, c):
        r = np.abs(x - c)
        if self.__rbf_equation == 'gaussian':
            return np.exp(-(self.__eps * r) ** 2)  # Gaussian
        elif self.__rbf_equation == 'multiquadric':
            return np.sqrt(1 + (self.__eps * r) ** 2)  # Multiquadric
        elif self.__rbf_equation == 'inv-quadric':
            return 1.0 / (1 + (self.__eps * r) ** 2)
        elif self.__rbf_equation == 'inv-multiquadric':
            return 1.0 / np.sqrt(1 + (self.__eps * r) ** 2)
        elif self.__rbf_equation == 'thin-plate':
            return r ** 2 * np.log(r)
        elif self.__rbf_equation == 'linear':
            return r
        else:
            print('Error! Not found function=', self.__rbf_equation)
    
    def __weight_solve(self, x, y):
        num = len(x)
        rbf_matrix = np.zeros((num, num))
        for i in range(num):
            for j in range(num):
                rbf_matrix[i, j] = self.__radial_basis_function(x[i], x[j])
        if self.__mode == 'interpolate':
            weights = np.linalg.solve(rbf_matrix, y)
        elif self.__mode == 'fitting':
            weights = np.linalg.solve(rbf_matrix.transpose() * rbf_matrix + self.__lambda * np.identity(num), rbf_matrix.transpose() * y)
        else:
            print('Error! not found interpolate mode=', self.__mode)
            sys.exit()
        return weights

    
    def __init__(self, x_array, y_array, mode='interpolate', kind='gaussian', eps=1.0, lamda=0.01):
        # Error handle
        if len(x_array) != len(y_array):
            print('Error! x-array and y-array must be same length')
            sys.exit()
        self.__x_sample = x_array
        self.__y_sample = y_array
        self.__mode = mode
        self.__rbf_equation = kind
        self.__eps = eps
        self.__lambda = lamda
        self.__weights = self.__weight_solve(self.__x_sample, self.__y_sample)
        ## plot RBFs
        # for i in range(len(self.__x_sample)):
        #     x = self.__x_sample[i]
        #     w = self.__weights[i]
        #     x_phi = np.arange(np.min(self.__x_sample), np.max(self.__x_sample), 0.01)
        #     y_phi = w * self.__radial_basis_function(x_phi, x)
        #     plt.plot(x_phi, y_phi)

    def __interpolate(self, x_input):
        rbf = self.__radial_basis_function(x_input, self.__x_sample)
        y_inter = np.sum(self.__weights * rbf)
        return y_inter

    def __call__(self, x):
        if type(x) is str:
            return self.__interpolate(float(x))
        elif type(x) in (int, float):
            return self.__interpolate(x)            
        elif type(x) is list or type(x).__module__ == np.__name__:
            return np.array([self.__interpolate(x_input) for x_input in x])
        else:
            print('Error! must be calulatable object')
            print('input:', type(x))
            sys.exit()
