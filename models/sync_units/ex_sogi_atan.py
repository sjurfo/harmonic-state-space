from sympy import symbols, cos, atan2, Matrix
from hsslib.hss import HSS


class SogiAtan(HSS):

    def setup(self):
        # States
        ua, ub = symbols('ua ub')
        self.x = [ua, ub]
        self.x0 = Matrix([cos(self.w0 * self.t),
                          cos(self.w0 * self.t)])

        # Inputs
        dtheta = symbols('dtheta')
        self.u = [dtheta]
        self.u0 = [0 * self.t]

        # Parameters
        k_sogi = symbols('k_sogi')
        self.p = [k_sogi]
        self.p_value = [1.414]

        # Algebraic equations
        ug = cos(self.w0 * self.t + dtheta)
        th = atan2(ub,ua)

        # Differential equations
        fa = self.w0*(k_sogi * (ug - ua) - ub)
        fb = ua * self.w0
        self.f = [fa, fb]

        # Outputs
        self.y = [th]


sogi = SogiAtan(N=3)
sogi.find_pss()

# Compute and plot eigenvalues
#sogi.calc_modal_props()
#import matplotlib.pyplot as plt
#fig, ax = plt.subplots()
#sogi.pss.modal_props.plot_eigs(ax)

# Do parametric sweep for sogi gain
#import numpy as np
#from parametric_studies import ParametricSweep

#sweep = ParametricSweep(sogi)
#sweep.eigenloci(np.linspace(0.5,2.5,21))
#sweep.eigenloci_plot()
