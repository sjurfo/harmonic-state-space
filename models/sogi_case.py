import numpy as np
from sympy import symbols, cos, sin, Matrix
from hss import HSS
from htf import HTF
from parametric_studies import ParametricSweep


class PLL(HSS):

    def setup(self):
        # States
        xpll, xd = symbols('xpll xd')
        self.x = [xpll, xd]

        # Inputs
        up = symbols('up')
        self.u = [up]
        self.u0 = [0 * self.t]

        # Parameters
        a_pll = symbols('a_pll')
        self.p = [a_pll]
        self.p_value = [60]

        # Fixed Parameters

        kp = 2 * a_pll
        ki = 2 * a_pll ** 2
        th_pll = xd + self.w0 * self.t

        uq = up-th_pll

        self.x0 = Matrix([0, 0])

        # Differential equations independent of FFP
        fpll = ki * uq
        fd = xpll + kp * uq

        self.f = [fpll, fd]

        # Outputs
        self.y = [xd]


class SogiPllT2(HSS):

    def setup(self):
        # States
        xa, xb, xpll, xd = symbols('xa xb xpll xd')
        self.x = [xa, xb, xpll, xd]
        self.x0 = Matrix([cos(self.w0 * self.t), sin(self.w0 * self.t),
                          0, 0])

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
        # Uncomment the following line to overwrite fundamental frequency
        # self.w0 = 100*np.pi

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


pll = PLL()
pll.find_pss()
param_sweep = ParametricSweep(pll)
param_sweep.eigenloci(np.linspace(20,150,30))
param_sweep.eigenloci_plot()



sogi = SogiPllT2()
sogi.find_pss()
param_sweep = ParametricSweep(sogi)
param_sweep.sweep([np.linspace(0.2,5,30), np.linspace(20,150,30)])
param_sweep.weakest_damping_3d()


pll = SogiPllT2()
import matplotlib.pyplot as plt
pll.find_pss()
pll.calc_modal_props()
pll.toep_BCD()
freqs = np.linspace(10,110,601)
#freqs = 1*np.logspace(0,3,500)

Hpll = HTF(pll, freqs)
midcol = Hpll.get_mid_col(tol=1e-10)
midcol.plot()
