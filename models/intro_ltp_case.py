import matplotlib.pyplot as plt
from sympy import symbols, cos, sin, atan2, Matrix
from hss import HSS


class IntroCase(HSS):

    def setup(self):
        # States
        x1, x2 = symbols('x1 x2')
        self.x = [x1, x2]

        x10 = 1e-15*sin(self.w0*self.t)
        x20 = 1e-15*sin(self.w0*self.t)
        self.x0 = Matrix([0*cos(self.w0 * self.t),
                          0*cos(self.w0 * self.t)])

        # Inputs
        u_i = symbols('u_i')
        self.u = [u_i]
        self.u0 = [0 * self.t]

        # Parameters
        q = symbols('q')
        self.p = [q]
        self.p_value = [1]

        a = 1
        mu = 0.99
        # Algebraic equations

        # Differential equations
        fx1 = x2*self.w0
        fx2 = self.w0*2*(-(a-q*cos(self.w0*self.t))*x1-mu*x2)
        self.f = [fx1,fx2]

        # Outputs
        self.y = [x1]

case = IntroCase(N=13)
case.find_pss()

import numpy as np
from parametric_studies import ParametricSweep

sweep = ParametricSweep(case)
sweep.eigenloci(np.linspace(0.,1.7,51))
sweep.eigenloci_plot()
