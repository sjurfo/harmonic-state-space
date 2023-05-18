import numpy as np
from sympy import symbols, cos, sin, atan2, Matrix
from hss import HSS
from htf import HTF
from parametric_studies import ParametricSweep


class SOGI_atan(HSS):

    def setup(self):
        # States
        ua, ub = symbols('ua ub')
        self.x = [ua, ub]
        self.x0 = Matrix([cos(self.w0 * self.t), sin(self.w0 * self.t)])

        # Inputs
        dtheta = symbols('dtheta')
        self.u = [dtheta]
        self.u0 = [0 * self.t]

        # Parameters
        ksog = symbols('ksog')
        self.p = [ksog]
        self.p_value = [2]

        # Fixed Parameters
        # Uncomment the following line to overwrite fundamental frequency
        # self.w0 = 100*np.pi

        # Algebraic equations
        ug = cos(self.w0 * self.t + dtheta)
        th = atan2(ub,ua)

        # Differential equations
        fa = ksog * (ug - ua) * self.w0 - ub * self.w0
        fb = ua * self.w0
        self.f = [fa, fb]

        # Outputs
        self.y = [th]


sogi = SOGI_atan(N=1)
sogi.find_pss()