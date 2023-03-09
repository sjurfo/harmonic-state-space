import numpy as np
from sympy import symbols, cos, sin, Matrix, solve, simplify
from hss import HSS
from htf import HTF
from parametric_studies import ParametricSweep


class SogiPll(HSS):

    def setup(self):
        # States
        xa, xb, xpll, xd = symbols('xa xb xpll xd')
        self.x = [xa, xb, xpll, xd]

        # Inputs
        up = symbols('up')
        self.u = [up]
        self.u0 = [0 * self.t]

        # Parameters
        ksog, a_pll = symbols('ksog a_pll')
        self.p = [ksog, a_pll]
        self.p_value = [2, 110]

        # Fixed Parameters
        ug = cos(self.w0 * self.t + up)
        # Uncomment the following line to overwrite fundamental frequency
        # self.w0 = 100*np.pi

        kp = 2 * a_pll
        ki = 2 * a_pll ** 2
        th_pll = xd + self.w0 * self.t
        ffp1 = True  # Integrator is before multiplication with omega
        ffp2 = True  # Integrator is before multiplication with omega
        # TODO: support for algebraic loops (in a DAE framework)
        # Solve loop symbolically
        wpll = symbols('wpll')
        uq = symbols('uq')
        # Different frequency feedback paths
        u_alpha = xa
        xa0 = cos(self.w0 * self.t)
        u_beta = xb
        xb0 = sin(self.w0 * self.t)
        if ffp1:
            u_alpha = xa * wpll
            xa0 = cos(self.w0 * self.t) / self.w0
        if ffp2:
            u_beta = xb * wpll
            xb0 = sin(self.w0 * self.t) / self.w0

        uq = simplify(
            solve(uq - (-sin(th_pll) * u_alpha + cos(th_pll) * u_beta).subs(wpll, xpll + self.w0 + kp * uq), uq)[0])
        wpll = xpll + self.w0 + kp * uq

        # Repeat frequency feedback paths as wpll is still a variable in u_alpha and u_beta
        u_alpha = xa
        u_beta = xb
        if ffp1:
            u_alpha = xa * wpll
        if ffp2:
            u_beta = xb * wpll

        # Check FFP here too
        fa = ksog * (ug - u_alpha) - u_beta
        fb = u_alpha

        if not ffp1:
            fa = (ksog * (ug - u_alpha) - u_beta) * wpll
        if not ffp2:
            fb = u_alpha * wpll

        self.x0 = Matrix([xa0, xb0, 0, 0])

        # Differential equations independent of FFP
        fpll = ki * uq
        fd = xpll + kp * uq

        self.f = [fa, fb, fpll, fd]

        # Outputs
        self.y = [xd]


class SogiFll(HSS):

    def setup(self):
        # States
        xa, xb, xfll = symbols('xa xb xfll')
        self.x = [xa, xb, xfll]

        # Inputs
        up = symbols('up')
        self.u = [up]
        self.u0 = [0 * self.t]

        # Parameters
        ksog, a_fll = symbols('ksog a_fll')
        self.p = [ksog, a_fll]
        self.p_value = [2, 110]

        # Fixed Parameters
        ug = cos(self.w0 * self.t + up)
        # Uncomment the following line to overwrite fundamental frequency
        # self.w0 = 100*np.pi

        # Algebraic variables
        w_fll = xfll + self.w0

        # Different frequency feedback paths
        ffp1 = False  # Integrator is before multiplication with omega (in-phase path)
        ffp2 = False  # Integrator is before multiplication with omega (in-quadrature path)

        u_alpha = xa
        xa0 = cos(self.w0 * self.t)
        u_beta = xb
        xb0 = sin(self.w0 * self.t)
        if ffp1:
            u_alpha = xa * w_fll
            xa0 = cos(self.w0 * self.t) / self.w0
        if ffp2:
            u_beta = xb * w_fll
            xb0 = sin(self.w0 * self.t) / self.w0

        kff = -a_fll * w_fll * ksog / (u_alpha ** 2 + u_beta ** 2)
        fxfll = (ug - u_alpha) * u_beta * kff

        # Check FFP here too
        fa = ksog * (ug - u_alpha) - u_beta
        fb = u_alpha

        if not ffp1:
            fa = (ksog * (ug - u_alpha) - u_beta) * w_fll
        if not ffp2:
            fb = u_alpha * w_fll

        self.x0 = Matrix([xa0, xb0, 0])
        self.f = [fa, fb, fxfll]

        # Outputs
        self.y = [w_fll]


sogi = SogiFll()
sogi.find_pss()
param_sweep = ParametricSweep(sogi)
param_sweep.sweep([np.flip(np.linspace(0.2, 5, 30)), np.flip(np.linspace(20, 150, 30))])
#ax = param_sweep.plot_parametric_study3d(offset=-200)
import matplotlib.pyplot as plt
from matplotlib import cm

fig, ax = plt.subplots(1)
levels = np.linspace(-200,0,21)

contourf = param_sweep.weakest_damping_contourf(ax, levels=levels, extend='both', cmap=cm.coolwarm)
param_sweep.weakest_damping_hatches(ax, levels=[-150, -20, 0, 1e10], colors='none', hatches=[None, '/', '++'])

cbar = fig.colorbar(contourf, ticks=np.linspace(levels[0], levels[-1], 5))

ax.set_xlabel(r'$k_{sog}$',fontsize=15)
ax.set_ylabel(r'$\alpha_{fll}$',fontsize=15)
#ax.set_zlabel(r'$Re[\lambda]$',fontsize=15)
ax.set_xticks([0.2, 2.5, 5])
ax.set_yticks([20, 50, 100,150])
#ax.set_zticks([-200,-100,0])
ax.tick_params(labelsize=12)
#ax.view_init(10,120)
plt.savefig("sogi_fll_2d_00.svg", bbox_inches="tight",
            pad_inches=0.3, transparent=True, format='svg', dpi=600)

