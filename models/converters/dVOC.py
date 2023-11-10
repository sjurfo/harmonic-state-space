import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, sin, cos, Matrix, Function

from hsslib.hss import HSS
from hsslib.complex_utils import Rotate2d, c2p


class dVOC_Statcom(HSS):
    def setup(self):
        """
        This is the dVOC-STATCOM from [1]
        [1]
        """
        # States
        ia = symbols('ia')  # filter current
        udc = symbols('udc')  # dc voltage
        vd, vq = symbols('vd vq')  # dVOC
        xa, xb = symbols('xa xb')  # SOGI
        xa2, xb2 = symbols('xa2 xb2')  # DC notch for compensation
        xdc = symbols('xdc')  # DCv control

        self.x = [ia, vd, vq, xa, xb, udc, xdc, xa2, xb2]

        # Input
        uin = symbols('uin')  # HSS input
        self.u = [uin]  # Inputs
        self.u0 = [0]  # Initial conditions u

        # Circuit
        Vg = 0.95
        thgrid = 0 #np.pi/2
        ug = Vg*np.sqrt(2)*sin(self.w0*self.t+thgrid)

        Cdc = 0.2e-3
        rf = 0.006
        lf = 0.052
        vdcb = 0.32
        sb = 1e-3
        cdc = Cdc * vdcb** 2 / sb  # Not per unit, just to simplify diff eq

        eta = symbols('eta')  # eta gives the P-w droop, dw=eta*dP  in per unit.
        mu = symbols('mu')  # mu gives the Q-V droop, dV=mu*dQ  in per unit
        rvir = symbols('rvir') # DC compensation

        # Parametric study
        self.p =       [rvir,  eta,  mu]    # Declare symbolic variables for parametric study
        self.p_value = [0.065, 0.05, 0.01]  # Values if they are not swept

        #  Constant control parameters
        ksog = np.sqrt(2)
        kpdc = 3
        Tdc = 0.01
        qref = 0.0
        vref = 1
        kidc = kpdc ** 2 * np.sqrt(2)
        udcref = 1.0

        # Initial conditions x (guess)
        I = 1
        ia0 = -I * np.sqrt(2) * cos(self.w0 * self.t+thgrid)
        xa0 = ia0/np.sqrt(2)
        xb0 = I * sin(self.w0 * self.t +thgrid)
        udc0 = udcref
        vd0 = vref*sin(thgrid)
        vq0 = -vref*cos(thgrid)

        self.x0 = Matrix([ia0, vd0, vq0, xa0, xb0, udc0, 0, 0, 0])

        # Reference frame
        th = self.w0*self.t
        yd, yq = Rotate2d(xa, xb, -th)
        # DAEs
        # Algebraic
        # DC control
        edc = -1 * (udcref ** 2 - udc ** 2)
        pref = (xdc + kpdc * edc)
        # dVOC inputs
        unorm = vd ** 2 + vq ** 2
        idref = 1 / unorm * (pref*vd+qref*vq)
        iqref = 1 / unorm * (pref*vq-qref*vd)
        e_id = idref-yd
        e_iq = iqref-yq
        va, vb = Rotate2d(vd, vq, th)
        mod = va*np.sqrt(2)-rvir*ia
        #mod = mod/udc  # CM mode
        mod = mod/(udcref+xa2)
        uc = mod*udc

        phi = (1 - unorm / vref ** 2)/2

        # Differential equations
        # Circuit
        fia = 1/(lf/self.w0)*(uc - rf*ia - uin - ug)
        fudc = -1 / cdc * (mod * ia)
        # DC
        fxdc = kidc * edc
        # VOC
        fvd = eta * self.w0 * (1 / mu * phi * vd - e_iq)
        fvq = eta * self.w0 * (1 / mu * phi * vq + e_id)
        # SOGI
        fxa = (ksog * (ia/np.sqrt(2) - xa) - xb) * self.w0
        fxb = xa * self.w0
        # modulation
        fxa2 = (1 / (Tdc * 2 * self.w0) * ((udc - 1) - xa2) - xb2) * 2 * self.w0
        fxb2 = xa2 * 2 * self.w0

        self.f = [fia, fvd, fvq, fxa, fxb, fudc, fxdc, fxa2, fxb2]

        # Output
        self.y = [-ia]


    def change_of_coordinates(self):
        self.h = [x for x in self.x]
        self.h[1], self.h[2] = c2p(self.x[1], self.x[2], 0)
        self.x[1].name = 'V'
        self.x[2].name = 'th'

        self.h[3], self.h[4] = c2p(self.x[3], self.x[4],-self.w0*self.t)
        self.x[3].name = 'Isog'
        self.x[4].name = 'th_sog'
        self.change_of_variable()


dvoc = dVOC_Statcom(N=13)
dvoc.find_pss()
#dvoc.change_of_coordinates()
#fig, ax = plt.subplots()
#dvoc.pss.plot_states(ax)

"""
# Sweep
from hsslib.parametric_studies import ParametricSweep
sweep = ParametricSweep(dvoc)
sweep.eigenloci(np.linspace(0.005,0.10), p_idx=2)
sweep.eigenloci_plot()
"""

# Compute HTF
"""
from hsslib.htf import HTF
dvoc.toep_BCD()

freqs = 1*np.logspace(0,3,500)
H = HTF(dvoc, freqs)
midcol = H.get_mid_col(tol=0.5)
midcol.plot(xscale='log', yscale='log')
"""