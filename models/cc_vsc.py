import matplotlib.pyplot as plt
from sympy import symbols, cos, sin, atan2, Matrix
from hss import HSS
from htf import HTF
from numpy import ix_, round
from numpy.linalg import svd
import numpy as np

def max_thd_htf(Y0, n):
    y_red = Y0[ix_(n,n)]
    u, s, vh = svd(y_red)
    u = round(u, 3)
    vh = round(vh, 3)
    return u, s, vh


class vsc_cc_1ph(HSS):

    def setup(self):

        # Model parameters
        l = 0.05
        r = 0.005
        id_ref = 0.8
        iq_ref = 0.7
        ki = 0.1
        kp = 10

        # States
        ua, ub = symbols('ua ub')
        ia = symbols('ia')
        x1, x2 = symbols('x1 x2')
        self.x = [ia, x1, x2, ua, ub]
        self.x0 = Matrix([0*self.t, 0*self.t, 0*self.t, cos(self.w0 * self.t),
                          sin(self.w0 * self.t)])

        # Inputs
        up = symbols('up')
        self.u = [up]
        self.u0 = [0 * self.t]

        # Parameters
        k_sogi = symbols('k_sogi')
        self.p = [k_sogi]
        self.p_value = [2]

        # Algebraic equations
        ug = cos(self.w0 * self.t) + up
        th = atan2(ub,ua)
        dia = id_ref*cos(th)-iq_ref*sin(th)-ia
        uc = x2 + kp*dia

        # Differential equations
        fi = 1/l*(uc-ug)-ia*r
        fx1 = self.w0*(ki * dia - x2)
        fx2 = x1 * self.w0
        fa = self.w0*(k_sogi * (ug - ua) - ub)
        fb = ua * self.w0
        self.f = [fi, fx1, fx2, fa, fb]

        # Outputs
        self.y = [-ia]


ccvsc = vsc_cc_1ph(N=9)
ccvsc.find_pss()
#fig, ax = plt.subplots()
#ccvsc.pss.plot_states(ax=ax)
ccvsc.toep_BCD()
Y = HTF(ccvsc, np.linspace(-1,1,2))
Y0 = np.round(Y.htf[1],5)
n1 = np.array([-5,-4,-3,-2,2,3,4,5])+9
n2 = np.array([-5,-3,3,5])+9
y_red = Y0[ix_(n1,n2)]
u, s, vh = svd(y_red)
u = round(u, 3)
vh = round(vh, 3)
#u, s, vh = max_thd_htf(Y0, np.array([-5,-3,-1,1,3,5])+9)
"""
import numpy as np
from parametric_studies import ParametricSweep

sweep = ParametricSweep(sogi)
sweep.eigenloci(np.linspace(0.5,2.5,21))
sweep.eigenloci_plot()
"""
