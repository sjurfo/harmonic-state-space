import matplotlib.pyplot as plt
from sympy import symbols, cos, sin, Matrix
import numpy as np
from matplotlib import cm

from hss import HSS
from parametric_studies import ParametricSweep


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
        self.p_value = [0.5, 0.5]

        # Differential equations
        fx1 = self.w0*x2
        fx2 = self.w0*(-2*zeta*x2-x1+beta*x1**3+cos(self.w0*self.t))
        self.f = [fx1,fx2]

        self.y = [x2]

duff = SofteningDuffing(N=15)
duff.find_pss()

fig, ax = plt.subplots()
duff.pss.plot_states(ax)

sweep = ParametricSweep(duff)
sweep.two_param_sweep([np.linspace(0.3,2, 25), np.linspace(0.1,4,25)])

fig, ax = plt.subplots(1)
levels = np.linspace(-100,0,21)

contourf = sweep.weakest_damping_contourf(ax, levels=levels, extend='both', cmap=cm.coolwarm)
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
