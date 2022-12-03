
import scipy.optimize as opt
import numpy as np

# create a class for data envelopment analysis with CCR model
class Data_Env_Analysis:

    def __init__(self, X, Y):
        self.X = X # input data matrix
        self.Y = Y # output data matrix
        self.n = X.shape[0] # number of observations
        self.m = X.shape[1] # number of inputs
        self.p = Y.shape[1] # number of outputs
        self.u= range(self.m) # weight the outputs
        self.v= range(self.p) # weight the inputs


    # create a output based CCR model function
    def output_based_CCR(self, u, v):
        # u is the weight for the outputs
        # v is the weight for the inputs
        
        # create a inequality constraint
        pass
   
    def __constraints(self, weights, unit):
        # weights is the weight for the inputs and outputs
        # unit is the unit to compute for
        # create a inequality constraint
        # denominator is sum of inputs data * input weights
        denominator = np.dot(self.X, weights[0:self.m])
        # numerator is sum of outputs data * output weights
        numerator = np.dot(self.Y, weights[self.m:])
        efficiency = numerator/denominator
            


     
