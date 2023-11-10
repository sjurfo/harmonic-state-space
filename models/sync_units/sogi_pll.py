import numpy as np
from sympy import symbols, cos, sin, Matrix

from hsslib.hss import HSS
from hsslib.htf import HTF


class SogiPll(HSS):

    def setup(self):
        # States
        xa, xb, xpll, xd = symbols('xa xb xpll xd')
        self.x = [xa, xb, xpll, xd]
        self.x0 = Matrix([cos(self.w0 * self.t), sin(self.w0 * self.t), 0, 0])

        # Inputs
        up = symbols('up')
        self.u = [up]
        self.u0 = [0*self.t]

        # Parameters
        ksog, a_pll = symbols('ksog a_pll')
        self.p = [ksog, a_pll]
        self.p_value = [2, 110]

        # Fixed Parameters
        ug = cos(self.w0*self.t+up)

        # Algebraic equations
        kp = 2 * a_pll
        ki = 2 * a_pll ** 2
        th_pll = xd + self.w0 * self.t
        uq = -sin(th_pll) * xa + cos(th_pll) * xb
        wpll = xpll + self.w0 + kp * uq

        # Differential equations
        fa = ksog * (ug - xa) * wpll - xb * wpll
        fb = xa * wpll
        fpll = ki * uq
        fd = xpll + kp * uq
        self.f = [fa, fb, fpll, fd]

        # Outputs
        self.y = [xd]


pll = SogiPll()
pll.find_pss()
pll.toep_BCD()

freqs = np.linspace(10,110,601)
Hpll = HTF(pll, freqs)
midcol = Hpll.get_mid_col(tol=1e-10)
midcol.plot()
