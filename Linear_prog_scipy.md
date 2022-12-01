## PYTHON FOR INDUSTRIAL ENGINEERS

# Linear Programming with Python

## Exploring SciPy’s “linprog” function

![](https://miro.medium.com/max/700/1*-iBGlpH-5kz6KnTuJLAO_g.jpeg)

Image by Kim available at Unsplash

# Operations Research

Operations Research is a scientific approach for decision making that seeks for the best design and operation of a system, usually under conditions requiring the allocation of scarce resources. The scientific approach for decision making requires the use of one or more mathematical/optimization models (i.e. representations of the actual situation) to make the optimum decision.

An optimization model seeks to find the values of the ***decision variables*** that optimize (maximize or minimize) an ***objective function*** among the set of all values for the decision variables that satisfy the given  ***constraints*** . Its three main components are:

* Objective function: a function to be optimized (maximized or minimized)
* Decision variables: controllable variables that influence the performance of the system
* Constraints: set of restrictions (i.e. linear inequalities or equalities) of decision variables. A non-negativity constraint limits the decision variables to take positive values (e.g. you cannot produce negative number of items *x*1, *x*2 and *x*3).

The solution of the optimization model is called the  **optimal feasible solution** .

# Modeling Steps

Modeling accurately an operations research problem represents the most significant-and sometimes the most difficult-task. A wrong model will lead to a wrong solution, and thus, will not solve the original problem. The following steps should be performed by different team members with different areas of expertise to obtain an accurate and greater view of the model:

1. **Problem definition** : defining the scope of the project and identifying that the result is the identification of three elements: description of decision variables, determination of the objective and determination of the limitations (i.e. constraints).
2. **Model construction** : translating the problem definition into mathematical relationships.
3. **Model solution** : using a standard optimization algorithm. Upon obtaining a solution, a sensitivity analysis should be performed to find out the behavior of the solution due to changes in some of the parameters.
4. **Model validity** : checking if the model works as it was supposed to.
5. **Implementation** : translating the model and the results into the recommendation of a solution.

# Linear Programming

Linear programming (also referred as LP) is an operations research technique used when all the objectives and constraints are linear (in the variables) and when all the decision variables are  **continuous** . In hierarchy, linear programming could be considered as the easiest operations research technique.

Python’s [SciPy](https://www.scipy.org/) library contains the [*linprog *](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html)function to solve linear programming problems. While using  *linprog* , there are two considerations to be taken into account while writing the code:

* The problem must be formulated as a minimization problem
* The inequalities must be expressed as ≤

## Minimization Problem

Let’s consider the following minimization problem to be solved:

![](https://miro.medium.com/max/416/1*vbhG3x0PadmRzOMRHsuL8A.png)

Let’s take a look at the Python code:

```python

# Import required libraries
import numpy as np
from scipy.optimize import linprog

# Set the inequality constraints matrix
# Note: the inequality constraints must be in the form of <=
A = np.array([[-1, -1, -1], [-1, 2, 0], [0, 0, -1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]])

# Set the inequality constraints vector
b = np.array([-1000, 0, -340, 0, 0, 0])

# Set the coefficients of the linear objective function vector
c = np.array([10, 15, 25])

# Solve linear programming problem
res = linprog(c, A_ub=A, b_ub=b)

# Print results
print('Optimal value:', round(res.fun, ndigits=2),
      '\nx values:', res.x,
      '\nNumber of iterations performed:', res.nit,
      '\nStatus:', res.message)
```

**Results**

```bash

# Optimal value: 15100.0 
# x values: [6.59999996e+02 1.00009440e-07 3.40000000e+02] 
# Number of iterations performed: 7 
# Status: Optimization terminated successfully.
```

## Maximization Problem

Since the *linprog *function from Python’s SciPy library is programmed to solve minimization problems, it is necessary to perform a transformation to the original objective function. Every minimization problem can be transformed into a maximization problem my multiplying the coefficients of the objective function by -1 (i.e. by changing their signs).

Let’s consider the following maximization problem to be solved:

![](https://miro.medium.com/max/372/1*H6lJ1yAlrxwYfu4CtQdzZg.png)

Let’s take a look at the Python code:

```python

# Import required libraries
import numpy as np
from scipy.optimize import linprog

# Set the inequality constraints matrix
# Note: the inequality constraints must be in the form of <=
A = np.array([[1, 0], [2, 3], [1, 1], [-1, 0], [0, -1]])

# Set the inequality constraints vector
b = np.array([16, 19, 8, 0, 0])

# Set the coefficients of the linear objective function vector
# Note: when maximizing, change the signs of the c vector coefficient
c = np.array([-5, -7])

# Solve linear programming problem
res = linprog(c, A_ub=A, b_ub=b)

# Print results
print('Optimal value:', round(res.fun*-1, ndigits=2),
      '\nx values:', res.x,
      '\nNumber of iterations performed:', res.nit,
      '\nStatus:', res.message)
```

**Results**

```
# Optimal value: 46.0 
# x values: [5. 3.] 
# Number of iterations performed: 5 
# Status: Optimization terminated successfully.
```

# Concluding Thoughts

Linear programming represents a great optimization technique for better decision making. The [*linprog*](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html) function from Python’s [SciPy](https://www.scipy.org/) library allows to solve linear programming problems with just a few lines of code. While there are other free optimization software (e.g. GAMS, AMPL, TORA, LINDO), using the *linprog *function could save you a significant amount of time by not having to code the simplex algorithm from scratch and go over each operation until the optimim value is reached.

*— —*

*If you found this article useful, feel welcome to download my personal codes on *[*GitHub*](https://github.com/rsalaza4/Python-for-Industrial-Engineering/tree/master/Linear%20Programming)*. You can also email me directly at rsalaza4@binghamton.edu and find me on *[*LinkedIn*](https://www.linkedin.com/in/roberto-salazar-reyna/)*. Interested in learning more about data analytics, data science and machine learning applications in the engineering field? Explore my previous articles by visiting my Medium *[*profile*](https://robertosalazarr.medium.com/)*. Thanks for reading.*

[https://towardsdatascience.com/linear-programming-with-python-db7742b91cb]()
