import matplotlib.pyplot as plt
from sympy import symbols, cos, sin, Matrix
import numpy as np

from hsslib.hss import HSS
from hsslib.parametric_studies import ParametricSweep


class SofteningDuffing(HSS):

    def setup(self):
        # States
        x1,x2 = symbols('x1 x2')
        self.x = [x1, x2]
        self.x0 = Matrix([-cos(self.w0*self.t),sin(self.w0*self.t)])

        # Inputs
        u = symbols('u')
        self.u = [u]
        self.u0 = [0 * self.t]

        # Parameters
        zeta = symbols('zeta')
        beta = symbols('beta')
        self.p = [zeta, beta]
        self.p_value = [1, 1]

        # Differential equations
        fx1 = self.w0*x2
        fx2 = self.w0*(-2*zeta*x2-x1+beta*x1**3+cos(self.w0*self.t)+u)
        self.f = [fx1,fx2]

        self.y = [x1]


def figure_weakest_damping():
    duff = SofteningDuffing(N=15)
    duff.find_pss()

    duff.p_value = [0.5,0.5]

    sweep = ParametricSweep(duff)
    sweep.two_param_sweep([np.linspace(0.3,2, 25), np.linspace(0.1,4,25)])
    fig, ax = plt.subplots(1)
    levels = np.linspace(-100,0,21)

    contourf = sweep.weakest_damping_contourf(ax, levels=levels, extend='both', cmap=plt.cm.coolwarm)
    sweep.weakest_damping_contourf(ax, levels=[-10, -1, 0, 1e10], colors='none', hatches=[None, '/', '++'])  # Add hatches
    cbar = fig.colorbar(contourf, ticks=np.linspace(levels[0], levels[-1], 5))
    cbar.set_label(r'$Re(\lambda_{weak})$', fontsize=25)
    cbar.ax.tick_params(labelsize=20)

    ax.set_xlabel(r'$\zeta$',fontsize=25)
    ax.set_ylabel(r'$\beta$',fontsize=25)
    ax.tick_params(labelsize=20)
    plt.show()
    #plt.savefig("duffing.svg", bbox_inches="tight",
    #            pad_inches=0.3, transparent=True, format='svg', dpi=600)
    return fig


def duffing_htf():
    duff = SofteningDuffing(N=15)
    duff.p_value = [0.5,1]
    duff.find_pss()

    duff.calc_modal_props()
    duff.toep_BCD()

    freqs = np.linspace(-50,150,601)
    #freqs = 1*np.logspace(0,3,500)

    from hsslib.htf import HTF
    Hduff = HTF(duff, freqs)
    midcol = Hduff.get_mid_col(tol=1e-10)
    midcol.plot()

