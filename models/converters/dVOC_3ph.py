from hsslib.hss import HSS
from sympy import symbols, sin, cos, Matrix
from hsslib.complex_utils import Rotate2d, c2p


class dVOC(HSS):
    def setup(self):
        """
        """
        # States
        id, iq = symbols('id iq')  # filter current
        vd, vq = symbols('vd vq')  # dVOC
        self.x = [id, iq, vd, vq]

        # Input
        uin = symbols('uin')  # HSS input
        self.u = [uin]  # Inputs
        self.u0 = [0]  # Initial conditions u

        # Circuit
        Vg = 1
        thgrid = 0 #np.pi/2
        ugd = Vg*cos(thgrid)
        ugq = Vg*sin(thgrid)

        xf = 0.1
        rf = 0.03
        lf = xf / self.w0

        eta = symbols('eta')  # eta gives the P-w droop, dw=eta*dP  in per unit.
        mu = symbols('mu')  # mu gives the Q-V droop, dV=mu*dQ  in per unit
        rvir = symbols('rvir')  # virtual resistance

        # Parametric study
        self.p = [rvir, eta, mu]  # Declare symbolic variables for parametric study
        self.p_value = [0., 0.03, 0.05]  # Values if they are not swept

        #  Constant parameters
        #  dVoc
        qref = 0.0
        pref = 1
        vref = 1

        # Initial conditions x (guess)
        id0 = 0
        iq0 = 0
        vd0 = vref*cos(thgrid)
        vq0 = -vref*sin(thgrid)

        self.x0 = Matrix([id0, iq0, vd0, vq0])

        # Reference frame
        th = self.w0*self.t
        # dVOC inputs
        unorm = vd ** 2 + vq ** 2
        idref = 1 / unorm * (pref*vd+qref*vq)
        iqref = 1 / unorm * (pref*vq-qref*vd)
        e_id = idref-id
        e_iq = iqref-iq
        #mod = mod/udc  # CM mode
        ud = vd
        uq = vq

        phi = (1 - unorm / vref ** 2)/2
        #unorm = vref ** 2

        # Differential equations
        # Circuit
        fid = 1/lf*(ud - ugd - rf*id +self.w0*lf*iq - uin)
        fiq = 1/lf*(uq - ugq - rf*iq - self.w0*lf*id)

        # VOC
        fvd = eta * self.w0 * (1 / mu * phi * vd - e_iq)
        fvq = eta * self.w0 * (1 / mu * phi * vq + e_id)

        self.f = [fid,fiq, fvd, fvq]
        # Output
        self.y = [iq]


    def change_of_coordinates(self):
        self.h = [x for x in self.x]
        #self.h[1], self.h[2] = c2p(self.x[1], self.x[2], 0)
        #self.x[1].name = 'V'
        #self.x[2].name = 'th'
        self.h[1], self.h[2] = Rotate2d(self.x[1], self.x[2], self.w0*self.t)
        self.x[1].name = 'va'
        self.x[2].name = 'vb'

        self.h[3], self.h[4] = c2p(self.x[3], self.x[4],-self.w0*self.t)
        self.x[3].name = 'Isog'
        self.x[4].name = 'th_sog'
        self.change_of_variable()
