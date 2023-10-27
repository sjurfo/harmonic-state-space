from hsslib.hss import HSS
import numpy as np
from sympy import symbols, sin, cos, Matrix, Function
import matplotlib.pyplot as plt


class Rotate2d(Function):
    @classmethod
    def eval(cls, x1,x2, th):
        return x1*cos(th)-x2*sin(th), x1*sin(th)+x2*cos(th)


class ComplexMult(Function):
    @classmethod
    def eval(cls, x11,x12,x21,x22):
        return x11*x21-x12*x22, x11*x22+x12*x21



class dVOC(HSS):
    def __init__(self):
        self.w0 = 100*np.pi
        self.vb = 0.126
        self.sb = 1e-3
        self.ib = self.sb/self.vb
        self.zb = self.vb/self.ib
        super().__init__(13)

    def setup(self):
        """
        This is the STATCOM-dVOC used in [1]
        [1]
        :return:
        """
        # States
        ia = symbols('ia')  # filter current
        vd, vq = symbols('vd vq')  # dVOC
        xa, xb = symbols('xa xb')  # SOGI

        self.x = [ia, vd, vq, xa, xb]

        # Input
        uin = symbols('uin')  # HSS input
        self.u = [uin]  # Inputs
        self.u0 = [0]  # Initial conditions u


        # Circuit
        Vg = 0.95
        thgrid = 0 #np.pi/2
        ug = Vg*np.sqrt(2)*cos(self.w0*self.t+thgrid)

        xf = 0.1
        rf = 0.01
        lf = xf / self.w0

        eta = symbols('eta')  # eta gives the P-w droop, dw=eta*dP  in per unit.
        mu = symbols('mu')  # mu gives the Q-V droop, dV=mu*dQ  in per unit
        ksog = symbols('ksog')  # sogi gain

        # Parametric study
        self.p =       [eta,  mu, ksog]    # Declare symbolic variables for parametric study
        self.p_value = [0.016, 0.05, np.sqrt(2)]  # Values if they are not swept

        #  Constant parameters

        #  dVoc
        qref = 0.44
        vref = 1

        # Initial conditions x (guess)
        I = 0
        ia0 = -I * np.sqrt(2) * cos(self.w0 * self.t+thgrid)
        xa0 = ia0/np.sqrt(2)
        xb0 = I * sin(self.w0 * self.t +thgrid)
        vd0 = vref*cos(thgrid)
        vq0 = -vref*sin(thgrid)

        self.x0 = Matrix([ia0, vd0, vq0, xa0, xb0])

        # Reference frame
        th = self.w0*self.t
        yd, yq = Rotate2d(xa, xb, -th)

        pref = 1
        # dVOC inputs
        unorm = vd ** 2 + vq ** 2
        idref = 1 / unorm * (pref*vd+qref*vq)
        iqref = 1 / unorm * (pref*vq-qref*vd)
        e_id = idref-yd
        e_iq = iqref-yq
        va, vb = Rotate2d(vd, vq, th)
        mod = va*np.sqrt(2)
        #mod = mod/udc  # CM mode
        uc = mod

        phi = (1 - unorm / vref ** 2)/2
        #unorm = vref ** 2

        # Differential equations
        # Circuit
        fia = 1/lf*(uc - rf*ia - uin - ug)

        # VOC
        fvd = eta * self.w0 * (1 / mu * phi * vd - e_iq)
        fvq = eta * self.w0 * (1 / mu * phi * vq + e_id)

        # SOGI
        fxa = (ksog * (ia/np.sqrt(2) - xa) - xb) * self.w0
        fxb = xa * self.w0

        self.f = [fia, fvd, fvq, fxa, fxb]

        # Output
        self.y = [-ia]


class dVOC_Statcom(HSS):
    def __init__(self):
        self.w0 = 100*np.pi
        self.vb = 0.2
        self.vdcb = 0.32
        self.sb = 1e-3
        self.ib = self.sb/self.vb
        self.idcb = self.sb/self.vdcb
        self.zb = self.vb/self.ib
        super().__init__(13)

    def setup(self):
        """
        This is the STATCOM-dVOC used in [1]
        [1]
        :return:
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

        xf = 0.052
        Cdc = 0.2e-3
        rf = 0.006
        lf = xf / self.w0
        cdc = Cdc * self.vdcb ** 2 / self.sb

        eta = symbols('eta')  # eta gives the P-w droop, dw=eta*dP  in per unit.
        mu = symbols('mu')  # mu gives the Q-V droop, dV=mu*dQ  in per unit
        rvir = symbols('rvir') # DC compensation


        # Parametric study
        self.p =       [rvir,  eta,  mu]    # Declare symbolic variables for parametric study
        self.p_value = [0.065, 0.05, 0.01]  # Values if they are not swept

        #  Constant parameters
        ksog = np.sqrt(2)
        kpdc = 3
        Tdc = 0.01

        #  dVoc
        qref = 0.0
        vref = 1

        # DC
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
        # Rotate state variables
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
        #unorm = vref ** 2

        # Differential equations
        # Circuit
        fia = 1/lf*(uc - rf*ia - uin - ug)
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


dvoc = dVOC()
dvoc.find_pss()
fig, ax = plt.subplots()
dvoc.pss.plot_states(ax)
#sweep = ParametricSweep(dvoc)
#sweep.eigenloci(np.flip(np.linspace(0.1,5,30)), p_idx=2)
#dvoc.p_value[1] = 0.05

#sweep.eigenloci_plot()