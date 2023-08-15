import numpy as np
from sympy import symbols, cos, sin, Matrix, solve, simplify
import matplotlib.pyplot as plt
from matplotlib import cm

from hss import HSS
from htf import HTF
from parametric_studies import ParametricSweep


class SogiPll(HSS):
    def __init__(self, ffpX=True, ffpY=True, N=4, w0=100*np.pi):
        # Types of feedback, type XY
        # If False (0), integrator is after frequency feedback
        # If True (1), integrator is before frequency feedback
        # ffpX for u_alpha
        # ffpY for u_beta
        self.ffpX = ffpX
        self.ffpY = ffpY
        super(SogiPll, self).__init__(N, w0)

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

        # TODO: support for algebraic loops (in a DAE framework)
        # Solve loop symbolically
        wpll = symbols('wpll')
        uq = symbols('uq')
        # Different frequency feedback paths
        u_alpha = xa
        xa0 = cos(self.w0 * self.t)
        u_beta = xb
        xb0 = sin(self.w0 * self.t)
        if self.ffpX:
            u_alpha = xa * wpll
            xa0 = cos(self.w0 * self.t) / self.w0
        if self.ffpY:
            u_beta = xb * wpll
            xb0 = sin(self.w0 * self.t) / self.w0

        uq = simplify(
            solve(uq - (-sin(th_pll) * u_alpha + cos(th_pll) * u_beta).subs(wpll, xpll + self.w0 + kp * uq), uq)[0])
        wpll = xpll + self.w0 + kp * uq

        # Repeat frequency feedback paths as wpll is still a variable in u_alpha and u_beta
        u_alpha = xa
        u_beta = xb
        if self.ffpX:
            u_alpha = xa * wpll
        if self.ffpY:
            u_beta = xb * wpll

        # Check FFP here too
        fa = ksog * (ug - u_alpha) - u_beta
        fb = u_alpha

        if not self.ffpX:
            fa = (ksog * (ug - u_alpha) - u_beta) * wpll
        if not self.ffpY:
            fb = u_alpha * wpll

        self.x0 = Matrix([xa0, xb0, 0, 0])

        # Differential equations independent of FFP
        fpll = ki * uq
        fd = xpll + kp * uq

        self.f = [fa, fb, fpll, fd]

        # Outputs
        self.y = [xd]


class SogiFll(HSS):
    def __init__(self, ffpX=True, ffpY=True, N=4, w0=100 * np.pi):
        # Types of feedback, type XY
        # If False (0), integrator is after frequency feedback
        # If True (1), integrator is before frequency feedback
        # ffpX for u_alpha
        # ffpY for u_beta
        self.ffpX = ffpX
        self.ffpY = ffpY
        super(SogiFll, self).__init__(N, w0)
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

        u_alpha = xa
        xa0 = cos(self.w0 * self.t)
        u_beta = xb
        xb0 = sin(self.w0 * self.t)
        if self.ffpX:
            u_alpha = xa * w_fll
            xa0 = cos(self.w0 * self.t) / self.w0
        if self.ffpY:
            u_beta = xb * w_fll
            xb0 = sin(self.w0 * self.t) / self.w0

        kff = -a_fll * w_fll * ksog / (u_alpha ** 2 + u_beta ** 2)
        fxfll = (ug - u_alpha) * u_beta * kff

        # Check FFP here too
        fa = ksog * (ug - u_alpha) - u_beta
        fb = u_alpha

        if not self.ffpX:
            fa = (ksog * (ug - u_alpha) - u_beta) * w_fll
        if not self.ffpY:
            fb = u_alpha * w_fll

        self.x0 = Matrix([xa0, xb0, 0])
        self.f = [fa, fb, fxfll]

        # Outputs
        self.y = [w_fll]


def ch4_figure(sweep, x, y, name):
    ax = sweep.weakest_damping_3d(offset=-200)
    ax.set_xlabel(r'$k_{sog}$', fontsize=20)
    ax.set_ylabel(r'$\alpha_{fll}$', fontsize=20, labelpad=15)
    if name=='sogi_pll':
        ax.set_ylabel(r'$\alpha_{pll}$', fontsize=20, labelpad=15)
    ax.set_zlabel(r'$Re[\lambda]$', fontsize=20, labelpad=15)
    ax.set_xticks([0, 2, 5])
    ax.set_yticks([0, 50, 100, 150])
    # ax.set_zticks([-200,-100,0])
    ax.tick_params(labelsize=17)
    ax.view_init(20, 120)
    plt.savefig(f'{name}_{int(x)}{int(y)}_3d.svg', bbox_inches="tight",
                pad_inches=0.0, transparent=True, format='svg', dpi=600)

    # plot 2-D contour
    fig, ax = plt.subplots(1)
    levels = np.linspace(-200, 0, 21)

    contourf = sweep.weakest_damping_contourf(ax, levels=levels, extend='both', cmap=cm.coolwarm)
    sweep.weakest_damping_contourf(ax, levels=[-150, -20, 0, 1e10], colors='none', hatches=[None, '/', '++'])

    ax.set_xlabel(r'$k_{sog}$', fontsize=25)
    ax.set_xticks([0.2, 2.5, 5])
    ax.set_yticks([20, 50, 100, 150])
    ax.set_ylabel(r'$\alpha$', fontsize=25)
    ax.tick_params(labelsize=20)
    ax.tick_params(width=2, length=4)
    if x or y:
        ax.set_xticklabels([])
        ax.set_xlabel('')
    if name=='sogi_pll':
        ax.set_yticklabels([])
        ax.set_ylabel('')

    plt.savefig(f'{name}_{int(x)}{int(y)}_2d.svg', bbox_inches="tight",
                pad_inches=0.3, transparent=True, format='svg', dpi=600)


ffp = (True, False)
for x in ffp:
    for y in ffp:
        # Declare model and parametric sweep
        pll = SogiPll(N=8, ffpX=x, ffpY=y)
        sweep_pll = ParametricSweep(pll)
        fll = SogiFll(N=8, ffpX=x, ffpY=y)
        sweep_fll = ParametricSweep(fll)

        ## 2-D parametric sweep
        sweep_pll.two_param_sweep([np.flip(np.linspace(0.2, 5, 30)), np.linspace(20, 150, 30)])
        sweep_fll.two_param_sweep([np.flip(np.linspace(0.2, 5, 30)), np.linspace(20, 150, 30)])

        ch4_figure(sweep_pll, x, y, 'sogi_pll')
        ch4_figure(sweep_fll, x, y, 'sogi_fll')

#cbar = fig.colorbar(contourf, ticks=np.linspace(levels[0], levels[-1], 5))
#cbar.ax.tick_params(labelsize=20)



