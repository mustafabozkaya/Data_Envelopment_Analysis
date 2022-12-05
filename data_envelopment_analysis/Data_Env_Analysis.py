
# create a class for data envelopment analysis with CCR model

import pandas as pd
from itertools import product

import gurobipy as gp
from gurobipy import GRB


class DEA_CCR:
    def __init__(self, multidict, inattr, outattr, verbose=True, objective='max'):
        self.multidict = multidict  # input-output values for the line
        # input-output values for the line
        self.dmus, self.inputs, self.outputs = multidict

        self.inattr = inattr  # input attributes
        self.outattr = outattr  # output attributes
        self.verbose = verbose  # verbose mode
        self.objective = objective  # objective function

    def solve_DEA(self, target):

        # Create LP model
        model = gp.Model('DEA')  # create model object

        # Decision variables
        # output weights (w)
        wout = model.addVars(self.outattr, name="outputWeight")
        # input weights (v)
        win = model.addVars(self.inattr, name="inputWeight")

        if self.objective == 'max':
            # Constraints
            ratios = model.addConstrs((gp.quicksum(self.outputs[h][r]*wout[r] for r in self.outattr) - gp.quicksum(
                self.inputs[h][i]*win[i] for i in self.inattr) <= 0 for h in self.dmus), name='ratios')  # output ratio constraints

            # sum of mutliplicative weights and target DMU inputs = 1
            normalization = model.addConstr((gp.quicksum(
                self.inputs[target][i]*win[i] for i in self.inattr) == 1), name='normalization')

            # add input and output weights must be greater and equal to 0
            model.addConstrs((wout[r] >= 0 for r in self.outattr), name='wout')

            model.addConstrs((win[i] >= 0 for i in self.inattr), name='win')

            # Objective function (maximize profit)
            model.setObjective(gp.quicksum(
                self.outputs[target][r]*wout[r] for r in self.outattr), GRB.MAXIMIZE)
        else:
            # Constraints
            ratios = model.addConstrs((gp.quicksum(self.outputs[h][r]*wout[r] for r in self.outattr) - gp.quicksum(
                self.inputs[h][i]*win[i] for i in self.inattr) <= 0 for h in self.dmus), name='ratios')  # output ratio constraints

            # sum of mutliplicative weights and target DMU inputs = 1
            normalization = model.addConstr((gp.quicksum(
                self.outputs[target][i]*wout[i] for i in self.outattr) == 1), name='normalization')

            # add input and output weights must be greater and equal to 0
            model.addConstrs((wout[r] >= 0 for r in self.outattr), name='wout')

            model.addConstrs((win[i] >= 0 for i in self.inattr), name='win')

            # Objective function (maximize profit)
            model.setObjective(gp.quicksum(
                self.inputs[target][r]*win[r] for r in self.inattr), GRB.MINIMIZE)

        # Run optimization engine
        if not self.verbose:
            model.params.OutputFlag = 0

        model.optimize()

        # Print results
        print(
            f"\nThe efficiency of target DMU {target} is {round(model.objVal,3)}")

        print("__________________________________________________________________")
        print(f"The weights for the inputs are:")
        for i in self.inattr:
            print(f"For {i}: {round(win[i].x,3)} ")

        print("__________________________________________________________________")
        print(f"The weights for the outputs are")
        for r in self.outattr:
            print(f"For {r} is: {round(wout[r].x,3)} ")
        print("__________________________________________________________________\n\n")

        return model.objVal

    def _print_result(self):

        performance = {}
        for h in self.dmus:
            performance[h] = self.solve_DEA(h)

        print("__________________________________________________________________")
        print("The efficiency scores for each DMU are:")
        for h in self.dmus:
            print(f"For {h} is {round(performance[h],3)}")
        print("__________________________________________________________________\n\n")


if __name__ == "__main__":

    multidict = gp.multidict({
        'Line_1': [{'Electricity': 1, 'Gas': 8, 'Water': 10, 'Chemical': 12, 'Maintance': 8.5, 'Labor': 4},   {'Production_Amount': 2,    'Quality': 0.6}],
        'Line_2': [{'Electricity': 1, 'Gas': 6, 'Water': 20, 'Chemical': 30, 'Maintance': 9,   'Labor': 4.5}, {'Production_Amount': 2.3,  'Quality': 0.7}],
        'Line_3': [{'Electricity': 2, 'Gas': 3, 'Water': 40, 'Chemical': 40, 'Maintance': 2,   'Labor': 1.5}, {'Production_Amount': 0.8,  'Quality': 0.25}],
        'Line_4': [{'Electricity': 2, 'Gas': 9, 'Water': 20, 'Chemical': 25, 'Maintance': 10,  'Labor': 6},   {'Production_Amount': 2.6,  'Quality': 0.86}]
    })

    dea = DEA_CCR(multidict, ['Electricity', 'Gas', 'Water', 'Chemical', 'Maintance', 'Labor'], [
                  'Production_Amount', 'Quality'], verbose=True, objective='min')

    dea._print_result()
