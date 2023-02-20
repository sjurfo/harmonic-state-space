from hss import HSS
from sympy import symbols, sin, cos, Matrix, atan2, atan, Function
import numpy as np


class Rotate2d(Function):
    @classmethod
    def eval(cls, x1,x2, th):
        return x1*cos(th)-x2*sin(th), x1*sin(th)+x2*cos(th)


class SogiPll_T2(HSS):
    def __init__(self):
        self.w0 = 100*np.pi
        self.U0 = 1
        super().__init__(5)

    def setup(self):
        #print('Setup symbolic HSS')
        #a_pll = 50
        ksog = 1
        a_pll = 200
        #ksog = 4
        # States
        xa, xb, xpll, xd = symbols('xa xb xpll xd')
        self.x = [xa,xb,xpll,xd]

        # Inputs
        ug = symbols('ug')
        self.u = [ug]

        # Params for parametric studies
        #ksog, a_pll = symbols('ksog a_pll')
        #self.p = [ksog,a_pll]
        self.p = []
        #self.p_range = [(0.01,3,15), (1,150,10)]

        # Outputs
        self.y = [xd]

        # Fixed params
        #ksog = 1

         #ksog = np.sqrt(2)*1

        # Initial conditions u
        self.u0 = [self.U0*cos(self.w0*self.t)+0*self.U0*cos(self.w0*3*self.t)]

        # Initial conditions x (guess)
        self.x0 = Matrix([self.U0*cos(self.w0*self.t),
                   self.U0*sin(self.w0*self.t),
                   0, 0])

        # Internal expressions
        kp = 2 * a_pll / self.U0
        ki = 2 * a_pll ** 2 / self.U0
        kp = 125
        ki = 6500
        # wpll
        th_pll = xd + self.w0*self.t
        uq = -sin(th_pll) * xa + cos(th_pll) * xb
        wpll = xpll + self.w0 + kp * uq
        wpll = self.w0
        # Differential equations
        fa = ksog*(ug-xa)*wpll-xb*wpll
        fb = xa*wpll
        fpll = ki*uq
        fd = xpll+kp*uq
        self.f = [fa, fb, fpll, fd]

        #print('x: {}'.format(self.x))

        self.A = Matrix(self.f).jacobian(self.x)
        self.B = Matrix(self.f).jacobian(self.u)
        self.C = Matrix(self.y).jacobian(self.x)
        self.D = Matrix(self.y).jacobian(self.u)


class WeaklyTVFLL(HSS):
    def __init__(self):
        self.w0 = 100*np.pi
        self.U0 = 1
        super().__init__(8)

    def setup(self):
        #print('Setup symbolic HSS')

        # States
        xa, xb, xfll, x2 = symbols('xa xb xfll x2')
        self.x = [xa, xb, xfll, x2]

        # Inputs
        up = symbols('up')
        self.u = [up]

        # Params for parametric studies
        ksog, a_fll = symbols('ksog a_fll')
        #ksog, a_fll = symbols('ksog a_fll')
        #self.p = [ksog, a_fll]
        self.p = [a_fll]
        p1 = np.linspace(20,100,30)
        self.paramvals = [p1]
        #self.p_range = [(0.1, 4, 30), (1, 150, 30)]



        # Fixed params
        ksog = np.sqrt(2)
        #a_fll = 20
        Tf = 0.001
        uh = 0

        # ksog = np.sqrt(2)*1

        # Initial conditions u
        cos0 = cos(self.w0*self.t)
        sin0 = sin(self.w0*self.t)
        ug = cos0
        self.u0 = [0]

        # Initial conditions x (guess)
        self.x0 = Matrix([self.U0 * cos0, self.U0 * sin0+0.001, 0, 0])

        # Algebraic equations (substitutable)
        err = ug+up-xa
        wfll = self.w0 + xfll
        kff = - a_fll*self.w0*ksog#/(xa**2+wfll**2*xb**2)
        lam_w = -self.w0*a_fll
        lam1 = 1000

        # Outputs
        self.y = [wfll]

        # Differential equations
        fa = (ksog * err - xb) * wfll
        fb = xa * wfll
        fx2 = (err*xb+(-ksog*err+xb-sin0)*xa-lam1*x2)
        ffll = x2*lam_w

        self.f = [fa, fb, ffll, fx2]

        # print('x: {}'.format(self.x))

        self.A = Matrix(self.f).jacobian(self.x)
        self.B = Matrix(self.f).jacobian(self.u)
        self.C = Matrix(self.y).jacobian(self.x)
        self.D = Matrix(self.y).jacobian(self.u)


class SogiFll_T1(HSS):
    def __init__(self):
        self.w0 = 100*np.pi
        self.U0 = 1
        super().__init__(4)

    def setup(self):
        #print('Setup symbolic HSS')

        # States
        xa, xb, xfll = symbols('xa xb xfll')
        self.x = [xa, xb, xfll]

        # Inputs
        ug = symbols('ug')
        self.u = [ug]

        # Params for parametric studies
        #ksog, a_fll = symbols('ksog a_fll')
        #ksog, a_fll = symbols('ksog a_fll')
        #self.p = [ksog, a_fll]
        # self.p = []
        #self.p_range = [(0.1, 4, 30), (1, 150, 30)]



        # Fixed params
        ksog = np.sqrt(2)
        a_fll = 20
        uh = 0

        # ksog = np.sqrt(2)*1

        # Initial conditions u
        self.u0 = [self.U0 * cos(self.w0 * self.t) + uh * self.U0 * cos(self.w0 * 3 * self.t)]

        # Initial conditions x (guess)
        self.x0 = Matrix([self.U0 * cos(self.w0 * self.t),
                          self.U0 / self.w0 * sin(self.w0 * self.t),
                          0])

        # Internal expressions

        wfll = self.w0 + xfll
        kff = - a_fll*wfll*ksog/(xa**2+wfll**2*xb**2)

        # Outputs
        self.y = [wfll]

        # Differential equations
        fa = ksog * (ug - xa) * wfll - xb * wfll ** 2
        fb = xa
        ffll = (ug-xa)*wfll*xb*kff

        self.f = [fa, fb, ffll]

        # print('x: {}'.format(self.x))

        self.A = Matrix(self.f).jacobian(self.x)
        self.B = Matrix(self.f).jacobian(self.u)
        self.C = Matrix(self.y).jacobian(self.x)
        self.D = Matrix(self.y).jacobian(self.u)


class Vsc_1ph(HSS):
    def __init__(self):
        self.w0 = 100*np.pi
        self.U0 = 220
        super().__init__(5)

    def setup(self):
        #print('Setup symbolic HSS')

        # States
        # FLL
        xa, xb, xfll = symbols('xa xb xfll')

        # Current control
        x1, x2, x3, x4 = symbols('x1 x2 x3 x4')

        # Circuit
        ia = symbols('ia')
        #self.x = [ia, xa, xb, xfll]
        self.x = [ia, x1, x2, x3, x4, xa, xb, xfll]

        # Inputs
        ug = symbols('ug')
        u_ex = symbols('u_ex')
        self.u = [ug]


        # Fixed params
        Lf = 2.536e-3
        Lg = 2.536e-3*3
        bw_c = 200
        Rf = 1e-7
        kp_c = bw_c*2*Lf
        #kp_c = 9
        ki_c = 2*Lf*bw_c**2
        #ki_c = 600
        print('Current control:')
        print(ki_c)
        print(kp_c)
        ksog = 1
        uh = 0
        id0 = 10
        # ksog = np.sqrt(2)*1

        # Params for parametric studies
        #ksog, Lg = symbols('ksog Lg')
        a_fll = 50*ksog
        # ksog, a_fll = symbols('ksog a_fll')
        #self.p = [ksog, Lg]
        #self.p_range = [(0.1, 4, 20), (0.1*Lf, Lf * 10, 20)]

        # Initial conditions u

        xa0 = self.U0 * cos(self.w0 * self.t)
        xb0 = self.U0 / self.w0 * sin(self.w0 * self.t)
        xfll0 = 0
        ia0 = id0 * cos(self.w0 * self.t)

        self.u0 = [self.U0 * cos(self.w0 * self.t) + uh * self.U0 * cos(self.w0 * 3 * self.t)]

        # Initial conditions x (guess)
        self.x0 = Matrix([ia0, 0, 0, 0, 0, xa0, xb0, xfll0])

        # DAEs

        wfll = self.w0 + xfll
        kff = - a_fll*wfll*ksog/(xa**2+wfll**2*xb**2)
        th = atan2(xb*self.w0,xa)
        #th = self.w0 * self.t
        err_ia = id0*cos(th)-ia
        uc = err_ia*kp_c + x1*ki_c + x3*ki_c + self.U0*cos(self.w0 * self.t)

        # Outputs
        self.y = [uc]

        fia = 1/(Lf+Lg)*(uc-Rf*ia-ug)
        ua = ug+Lg*fia
        fa = ksog * (ua - xa) * wfll - xb * wfll ** 2
        fb = xa
        ffll = (ua-xa)*wfll*xb*kff

        fx1 = -self.w0*x2 + err_ia
        fx2 = self.w0*x1 #+ 0.000001*x2
        fx3 = -self.w0*3*x4 + err_ia
        fx4 = self.w0*3*x3 #+ 0.000001*x4

        #self.f = [fia, fa, fb, ffll]

        self.f = [fia, fx1, fx2, fx3, fx4, fa, fb, ffll]
        # print('x: {}'.format(self.x))

        self.A = Matrix(self.f).jacobian(self.x)
        self.B = Matrix(self.f).jacobian(self.u)
        self.C = Matrix(self.y).jacobian(self.x)
        self.D = Matrix(self.y).jacobian(self.u)


class Vsc_1ph_LCL(HSS):
    def __init__(self):
        self.w0 = 100*np.pi
        self.U0 = 220*np.sqrt(2)
        super().__init__(4)

    def setup(self):
        #print('Setup symbolic HSS')

        # States
        # FLL
        xa, xb, xfll = symbols('xa xb xfll')

        # Current control
        x1, x2, x3, x4 = symbols('x1 x2 x3 x4')

        # Circuit
        ia, i1, ucap = symbols('ia i1 ucap')

        self.x = [ia, i1, ucap, x1, x2, x3, x4, xa, xb, xfll]

        # Inputs
        ug = symbols('ug')
        u_ex = symbols('u_ex')
        self.u = [ug]

        # Outputs
        self.y = [-ia]

        # Fixed params
        L1 = 0.75e-3
        C1 = 6.8e-6
        L2 = 0.45e-3
        Lg = 10e-3
        bw_c = 200
        Rf = 1e-9
        kp_c = 9
        #ki_c = 2*Lf*bw_c**2
        ki_c = 600
        print('Current control:')
        print(ki_c)
        print(kp_c)
        kc_f = 0.6
        ksog = 0.5
        uh = 0
        id0 = 5e3/220
        print('id0: {}'.format(id0))

        # ksog = np.sqrt(2)*1

        # Params for parametric studies
        #ksog, Lg = symbols('ksog Lg')
        a_fll = 50*ksog
        # ksog, a_fll = symbols('ksog a_fll')
        #self.p = [ksog, Lg]
        #self.p_range = [(0.1, 4, 20), (0.1*Lf, Lf * 10, 20)]

        # Initial conditions u
        self.u0 = [self.U0 * cos(self.w0 * self.t) + uh * self.U0 * cos(self.w0 * 3 * self.t)]

        # Initial conditions x (guess)
        xa0 = self.U0 * cos(self.w0 * self.t)
        xb0 = self.U0 / self.w0 * sin(self.w0 * self.t)
        xfll0 = 0
        ia0 = id0 * cos(self.w0 * self.t)
        self.x0 = Matrix([ia0, ia0, xa0, 0, 0, 0, 0, xa0, xb0, xfll0])

        # DAEs

        wfll = self.w0 + xfll
        kff = - a_fll*wfll*ksog/(xa**2+wfll**2*xb**2)
        th = atan2(xb*self.w0, xa)
        err_ia = id0*cos(th)-ia
        fia = 1/(L2+Lg)*(ucap-Rf*ia-ug)
        ua = ug+Lg*fia
        fucap = 1/C1*(i1-ia)
        uc = self.U0*cos(self.w0*self.t)+x1*ki_c+err_ia*kp_c + x3*ki_c + kc_f*ua - 10*(i1-ia)
        fi1 = 1/(L1)*(uc-ucap)
        fa = ksog * (ua - xa) * wfll - xb * wfll ** 2
        fb = xa
        ffll = (ua-xa)*wfll*xb*kff

        fx1 = -self.w0*x2 + err_ia
        fx2 = self.w0*x1# + 0.000001*x2
        fx3 = -self.w0*3*x4 + err_ia
        fx4 = self.w0*3*x3# + 0.000001*x4

        self.f = [fia, fi1, fucap, fx1, fx2, fx3, fx4, fa, fb, ffll]

        # print('x: {}'.format(self.x))

        self.A = Matrix(self.f).jacobian(self.x)
        self.B = Matrix(self.f).jacobian(self.u)
        self.C = Matrix(self.y).jacobian(self.x)
        self.D = Matrix(self.y).jacobian(self.u)


class Resonator(HSS):
    def __init__(self):
        self.w0 = 100*np.pi
        self.U0 = 1
        super().__init__(4)

    def setup(self):
        #print('Setup symbolic HSS')
        Lf = 2.536e-3
        bw_c = 200
        Rf = 1e-7
        kp_c = bw_c * Lf
        kp_c = 9
        ki_c = 1 / (2 * Lf * bw_c ** 2)
        ki_c = 600
        # States
        # Current control
        x1, x2 = symbols('x1 x2')
        self.x = [x1,x2]

        # Inputs
        ierr = symbols('ierr')
        self.u = [ierr]

        # Fixed params

        # Initial conditions u
        self.u0 = [0]

        # Initial conditions x (guess)
        self.x0 = Matrix([0,0])

        # Internal expressions


        # Outputs
        uc = x1*ki_c+ierr*kp_c
        self.y = [uc]

        # Differential equations
        fx1 = -self.w0 * x2 + ierr
        fx2 = self.w0 * x1 + 0.0001*x2

        self.f = [fx1,fx2]

        # print('x: {}'.format(self.x))

        self.A = Matrix(self.f).jacobian(self.x)
        self.B = Matrix(self.f).jacobian(self.u)
        self.C = Matrix(self.y).jacobian(self.x)
        self.D = Matrix(self.y).jacobian(self.u)


class Sogi(HSS):
    def __init__(self):
        self.w0 = 100*np.pi
        self.U0 = 1
        super().__init__(5)

    def setup(self):

        ksog = 0.2

        # States
        xa, xb = symbols('xa xb')
        self.x = [xa,xb]

        # Inputs
        ug = symbols('ug')
        self.u = [ug]

        # Params for parametric studies
        ksog = symbols('ksog')
        self.p = [ksog]
        p1 = np.linspace(0.1, 5, 20)
        self.paramvals = [p1]

        # Outputs
        self.y = [xa]

        # Initial conditions u
        self.u0 = [self.U0*cos(self.w0*self.t)+0*self.U0*cos(self.w0*3*self.t)]

        # Initial conditions x (guess)
        self.x0 = Matrix([cos(self.w0*self.t),sin(self.w0*self.t)])

        # Internal expressions
        # wpll

        wpll = self.w0
        th = self.w0 * self.t  # +vocbus.angle
        # Algebraic equations
        #xa = xd
        #xb = xq
        # Differential equations
        fa = ksog*(ug-xa)*wpll-xb*wpll
        fb = xa*wpll

        self.f = [fa, fb]

        #print('x: {}'.format(self.x))

        self.A = Matrix(self.f).jacobian(self.x)
        self.B = Matrix(self.f).jacobian(self.u)
        self.C = Matrix(self.y).jacobian(self.x)
        self.D = Matrix(self.y).jacobian(self.u)


class Sogi_Dq(HSS):
    def __init__(self):
        self.w0 = 100*np.pi
        self.U0 = 1
        super().__init__(5)

    def setup(self):

        ksog = 6

        # States
        xd, xq = symbols('xd xq')
        self.x = [xd,xq]

        # Inputs
        ug = symbols('ug')
        self.u = [ug]

        # Params for parametric studies
        ksog = symbols('ksog')
        self.p = [ksog]
        p1 = np.linspace(0.1,5,20)
        self.paramvals = [p1]
        #self.p = []
        #self.p_range = [(0.01,3,15), (1,150,10)]

        # Outputs
        self.y = [xd]

        # Initial conditions u
        self.u0 = [self.U0*cos(self.w0*self.t)+0*self.U0*cos(self.w0*3*self.t)]

        # Initial conditions x (guess)
        self.x0 = Matrix([1,0])

        # Internal expressions
        # wpll

        wpll = self.w0
        th = self.w0 * self.t  # +vocbus.angle
        # Algebraic equations
        xa, xb = Rotate2d(xd,xq,th)
        #xa = xd
        #xb = xq
        # Differential equations
        fa = ksog*(ug-xa)*wpll-xb*wpll**2
        fb = xa

        fd, fq = Rotate2d(fa+self.w0*xb, fb-self.w0*xa,-th)
        self.f = [fd, fq]

        #print('x: {}'.format(self.x))

        self.A = Matrix(self.f).jacobian(self.x)
        self.B = Matrix(self.f).jacobian(self.u)
        self.C = Matrix(self.y).jacobian(self.x)
        self.D = Matrix(self.y).jacobian(self.u)

#s = Vsc_1ph_LCL()
s = WeaklyTVFLL()
#s.parametric_sweep()
#s.find_pss()
#s.calc_eigs()
pfs = s.eigenloci()
s.eigenloci_plot(pfs)
#print(s.eigen)
#print(s.weak_damp)
"""
s.toepBCD()
freqs = np.linspace(0,150,601)
#an_htfs, idxs = s.hss_to_htf(freqs, tol=2e-15)
htm = s.hss_to_htf(freqs, tol=2e-15)
an_htfs, idxs = htm.get_mid_col(tol=1e-10)
#freqs = np.linspace(*fr_sp)


import matplotlib.pyplot as plt

plt.rcParams.update({'font.family':'Comic Sans MS'})
plt.rcParams['text.usetex'] = False

fig, axs = plt.subplots(2)
axs[0].plot(freqs, np.abs(an_htfs[0]),label=r'$H_{-1}$')
axs[0].plot(freqs, np.abs(an_htfs[1]),label=r'$H_{1}$')
axs[1].plot(freqs, np.angle(an_htfs[0]),label=r'$H_{-1}$')
axs[1].plot(freqs, np.angle(an_htfs[1]),label=r'$H_{1}$')
axs[0].legend()
axs[1].set_xlabel("Frequency [Hz]")
axs[1].set_ylabel(r"$\angle H(j\omega)$")
axs[0].set_ylabel(r"$|H(j\omega)|$")
plt.show()
"""
"""
# Plot frequency-amplitude plane
fig, ax = plt.subplots(1)
[ax.plot(freqs-(s.N-n)*s.w0/(2*np.pi), np.abs(an_htfs[ind]), color='k') for ind, n in enumerate(idxs)]
ax.plot(freqs, 1*np.ones(len(freqs)), '--', linewidth=3, alpha=0.4, color='k')
#ax.vlines(freqs[150], ymax=2, ymin=-2, colors='grey', ls='--')
#ax.vlines(freqs[450], ymax=2, ymin=-2, colors='grey', ls='--')
#[ax.plot(freqs, np.abs(an_htfs[ind]), color='k') for ind, n in enumerate(idxs)]

#for i in range(0,501,50):
#    ax.vlines(freqs[i], ymax=2, ymin=-2, colors='grey', ls='--')
ax.set_xlabel(r'$f[Hz]$')
#ax.set_ylabel(r"$|H(j\omega)|$")
#ax.imshow(a, cmap='hot', interpolation='nearest')
plt.show()
"""
"""
# Plot spline visualization
fig, ax = plt.subplots(1)
[ax.plot(freqs, np.imag(an_htfs[ind]), color='k') for ind, n in enumerate(idxs)]
#for i in range(0,501,50):
#    ax.vlines(freqs[i], ymax=2, ymin=-2, colors='grey', ls='--')
#ax.set_xlabel(r'$f_{in}[Hz]$')
ax.set_xlabel(r'$time[s]$')
ax.set_ylabel(r"$Im\{H(j\omega)\}$")
#ax.imshow(a, cmap='hot', interpolation='nearest')
plt.show()

"""
#s2.parametric_sweep()
#import matplotlib.pyplot as plt
#s.plotstates()
#plt.figure()
#plt.scatter(np.real(s.eigen),np.imag(s.eigen))
#plt.show()
"""
#3-d plot:
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
[ax.plot(freqs, freqs-(s.N-n)*s.w0/(2*np.pi), abs(an_htfs[ind]), color='k') for ind, n in enumerate(idxs)]

ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 0.5, 0.7, 1]))
ax.plot(freqs, freqs, 1*np.ones(len(freqs)), '--', linewidth=3, alpha=0.4, color='k')

plt.rcParams['text.usetex'] = True
ax.set_xlabel(r'$time[s]$')
ax.set_ylabel(r'$f [Hz]$')
#ax.set_zlabel(r"$|H(j\omega)|$")
plt.show()
"""

"""

# Plot f-t projection
fig, ax = plt.subplots(1)
maxx = np.max(np.abs(an_htfs[0]))
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

for i in range(len(freqs)-1):
    for ind, n in enumerate(idxs):
        amp = np.abs(an_htfs[ind][i])
        ax.plot(freqs[i:i+2],freqs[i:i+2]-(s.N-n)*s.w0/(2*np.pi), linewidth = 2, color = mpl.cm.viridis(amp/maxx))
    if i%40==0:
        ax.plot(freqs[i], freqs[i], '+', color=mpl.cm.viridis(1/maxx), markersize = 10)

divider = make_axes_locatable(ax)
ax_cb = divider.new_horizontal(size="5%", pad=0.05)
cb1 = mpl.colorbar.ColorbarBase(ax_cb, cmap=mpl.cm.viridis, orientation='vertical', norm = mpl.colors.Normalize(vmin=0, vmax=maxx))
plt.gcf().add_axes(ax_cb)
inds = [i*50 for i in range(4)]
print(inds)
ax.set_xticks(inds)
ax.set_xlabel(r'$time[s]$')
ax.set_ylabel(r'$f\ \ [Hz]$')
plt.show()
"""
