"""
Data Envelopment Analysis implementation

Sources:
Sherman & Zhu (2006) Service Productivity Management, Improving Service Performance using Data Envelopment Analysis (DEA) [Chapter 2]
ISBN: 978-0-387-33211-6
http://deazone.com/en/resources/tutorial

"""

import numpy as np
from scipy.optimize import linprog  # Linear Programming (Simplex) Algorithm
from scipy.optimize import fmin_slsqp  # Sequential Least SQuares Programming
from scipy.optimize import fmin_bfgs  # Broyden-Fletcher-Goldfarb-Shanno


class DEA(object):

    def __init__(self, inputs, outputs):
        """
        Initialize the DEA object with input data
        num_record = number of entities (observations)
        num_in = number of inputs (variables, features)
        num_out= number of outputs
        :param inputs: inputs, n x m numpy array
        :param outputs: outputs, n x r numpy array
        :return: self
        """

        # supplied data
        self.inputs = inputs
        self.outputs = outputs

        # parameters
        self.num_record = inputs.shape[0]
        self.num_in = inputs.shape[1]
        self.num_out = outputs.shape[1]

        # iterators
        self.unit_ = range(self.num_record)
        self.input_ = range(self.num_in)
        self.output_ = range(self.num_out)

        # result arrays
        self.output_w = np.zeros(
            (self.num_out, 1), dtype=np.float64)  # output weights
        self.input_w = np.zeros(
            (self.num_in, 1), dtype=np.float64)  # input weights
        self.lambdas = np.zeros((self.num_record, 1),
                                dtype=np.float64)  # unit efficiencies
        self.efficiency = np.zeros_like(self.lambdas)  # thetas means efficiencies

        # names
        self.names = []
    def _objective(self):


    def __efficiency(self, unit):
        """
        Efficiency function with already computed weights
        :param unit: which unit to compute for
        :return: efficiency
        """

        # compute efficiency
        # denominator is sum of inputs data * input weights
        denominator = np.dot(self.inputs, self.input_w)
        # numerator is sum of outputs data * output weights
        numerator = np.dot(self.outputs, self.output_w)
        efficiency = numerator/denominator  # array of efficiencies
        # array of differences between numerator and denominator
        self.diff_unit = numerator-denominator

        return efficiency[unit]

    def __target(self, weights, unit):
        """
        Theta target function for one unit
        :param weights: combined weights (input, output, lambdas)
        :param unit: which production unit to compute
        :return: theta
        """
        in_w, out_w, lambdas = weights[:self.num_in], weights[self.num_in:(
            self.num_in+self.num_out)], weights[(self.num_in+self.num_out):]  # unroll the weights
        # denominator is sum of inputs data * input weights
        denominator = np.dot(self.inputs[unit], in_w)
        # numerator is sum of outputs data * output weights
        numerator = np.dot(self.outputs[unit], out_w)

        return numerator/denominator

    def __ieconstraints(self, weights, unit):
        """
        Constraints for optimization for one unit
        :param x: combined weights
        :param unit: which production unit to compute
        :return: array of constraints
        """

        in_w, out_w, lambdas = weights[:self.num_in], weights[self.num_in:(
            self.num_in+self.num_out)], weights[(self.num_in+self.num_out):]  # unroll the weights
        constr = []  # init the constraint array

        # for each input, lambdas with inputs
        for input in self.input_:
            # target function for unit with weights x
            t = self.__target(weights, unit)
            # left hand side of the constraint
            lhs = np.dot(self.inputs[:, input], lambdas)
            cons = t*self.inputs[unit, input] - lhs  # constraint
            constr.append(cons)  # add constraint to the array

        # for each output, lambdas with outputs
        for output in self.output_:
            # left hand side of the constraint
            lhs = np.dot(self.outputs[:, output], lambdas)
            # constraint (note the sign)
            cons = lhs - self.outputs[unit, output]
            constr.append(cons)  # add constraint to the array
        # for each unit
        for u in self.unit_:
            constr.append(lambdas[u])  # add constraint to the array

        return np.array(constr)

    def __econstraints(self, weights, unit,input_or_output):
       constr = []  # init the constraint array
       if input_or_output == 'input':
           in_w, out_w, lambdas = weights[:self.num_in], weights[self.num_in:(
               self.num_in+self.num_out)], weights[(self.num_in+self.num_out):]
              for input in self.input_:


    def __optimize(self):
        """
        Optimization of the DEA model
        Use: http://docs.scipy.org/doc/scipy-0.17.0/reference/generated/scipy.optimize.linprog.html
        A = coefficients in the constraints (matrix) (n x m)
        b = rhs of constraints (<=) (1 x n)
        c = coefficients of the target function (1 x m)
        :return:
        """
        d0 = self.num_in + self.num_out + self.num_record  # number of variables
        # iterate over units
        for unit in self.unit_:
            # weights
            # initial weights are random numbers between 0 and 1
            x0 = np.random.rand(d0)

            equals_const_list=[] # list of equality constraints
            # for each input
            for input in self.input_:
                # equality constraint
                equals_const_list.append({'type': 'eq', 'fun': lambda x: np.dot(self.inputs[:, input], x[self.num_in+self.num_out:]) - self.inputs[unit, input]})
            # minimize the target function
            x0 = fmin_slsqp(self.__target, x0,
                            f_ieqcons=self.__ieconstraints,
                            f_eqcons=self._econstraints,
                            args=(unit,)
                            )
            # unroll weights
            self.input_w, self.output_w, self.lambdas = x0[:self.num_in], x0[self.num_in:(
                self.num_in+self.num_out)], x0[(self.num_in+self.num_out):]
            self.efficiency[unit] = self.__efficiency(unit)

    def name_units(self, names):
        """
        Provide names for units for presentation purposes
        :param names: a list of names, equal in length to the number of units
        :return: nothing
        """

        assert(self.num_record == len(names))

        self.names = names

    def fit(self):
        """
        Optimize the dataset, generate basic table
        :return: table
        """

        self.__optimize()  # optimize

        print("Final efficient for each unit:\n")
        print("---------------------------\n")
        for n, eff in enumerate(self.efficiency):
            if len(self.names) > 0:
                name = "Unit %s" % self.names[n]
            else:
                name = "Unit %d" % (n+1)
            print("%s theta: %.4f" % (name, eff))
            print("\n")
        print("---------------------------\n")


if __name__ == "__main__":
    X = np.array([
        [20., 300., 50, 100],
        [30., 400., 5, 100],
        [40., 500., 50, 10],
        [50., 600., 5, 10],
        [60., 700., 50, 1],
        [70., 150., 5, 1],
        [80., 250., 50, 1],
        [90., 350., 5, 1],
        [100., 450., 50, 1],
        [110., 50., 5, 1],
    ])
    y = np.array([
        [100., 120],
        [200., 50],
        [50., 100],
        [100., 200],
        [200., 100],
        [100., 20],
        [200., 50],
        [50., 100],
        [10., 200],
        [200., 100],
    ])
    names = [
        'Bratislava',
        'Zilina',
        'Kosice',
        'Presov',
        'Poprad',
        'Banska Bystrica',
        'Nitra',
        'Tacna',
        'Trnava',
        "Alicante"
    ]
    dea = DEA(X, y)
    dea.name_units(names)
    dea.fit()
