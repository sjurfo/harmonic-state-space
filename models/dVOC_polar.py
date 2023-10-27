from hsslib.hss import HSS
import numpy as np
from sympy import symbols, sin, cos, Matrix, atan2, Function

from hsslib.parametric_studies import ParametricSweep



class ComplexMult(Function):
    @classmethod
    def eval(cls, x11,x12,x21,x22):
        return x11*x21-x12*x22, x11*x22+x12*x21

class Rotate2d(Function):
    @classmethod
    def eval(cls, x1,x2, th):
        return ComplexMult(x1, x2, cos(th), sin(th))


class c2p(Function):
    @classmethod
    def eval(cls, x,y, th0):
        """
        Cartesian to polar
        :param x: x axis coordinate
        :param y: y axis coordinate
        :param th0: reference angle
        :return: polar magnitude and angle
        """
        xd,xq = Rotate2d(x,y, -th0)
        return (x**2+y**2)**0.5, atan2(xq,xd)


class p2c(Function):
    @classmethod
    def eval(cls, v, th, th0):
        """
        Polar to cartesian
        :param v: Polar amplitude
        :param th: angle
        :param th0: reference angle
        :return: Cartesian coordinates x,y
        """
        return v*cos(th+th0), v*sin(th+th0)


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

        xf = 0.052
        rf = 0.006
        lf = xf / self.w0
        ksog = np.sqrt(2)

        eta = symbols('eta')  # eta gives the P-w droop, dw=eta*dP  in per unit.
        mu = symbols('mu')  # mu gives the Q-V droop, dV=mu*dQ  in per unit
        rvir = symbols('rvir')  # virtual resistance

        # Parametric study
        self.p = [rvir, eta, mu]  # Declare symbolic variables for parametric study
        self.p_value = [0.065, 0.05, 1000]  # Values if they are not swept

        #  Constant parameters

        #  dVoc
        qref = 0.0
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
        mod = va*np.sqrt(2)-rvir*ia
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


class SofteningDuffing(HSS):
    def __init__(self):
        super().__init__(13)
    def setup(self):
        # States
        x1,x2 = symbols('x1 x2')
        self.x = [x1, x2]
        th0 = self.w0*self.t
        self.x0 = Matrix([-cos(self.w0*self.t),sin(self.w0*self.t)])
        self.x0 = Matrix([sin(self.w0 * self.t), cos(self.w0 * self.t)])

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
        #fx1 = self.w0*x2
        #fx2 = self.w0*(-2*zeta*x2-x1+beta*x1**3+cos(self.w0*self.t)+u)
        fx1 = self.w0*(-2*zeta*x1-x2+beta*x2**3+sin(self.w0*self.t)+u)
        fx2 = self.w0*x1
        """
        V, th = symbols('V th')
        g = Matrix(c2p(x1,x2,th0))
        fold = Matrix([fx1,fx2])
        xold = Matrix([x1,x2])
        fnew = g.jacobian(xold)*fold+g.jacobian([self.t])
        h1,h2 = p2c(V,th,th0)
        fnew1 = fnew.subs({x1:h1, x2:h2})
        self.f = list(fnew1)
        self.x = [V, th]
        self.x0 = Matrix([0.5, 0.0])
        self.y = [V]
        """
        self.f = [fx1,fx2]
        self.y = [x1]

    def change_of_coordinates(self):
        self.h = [x for x in self.x]
        xa = self.x[0]
        xb = self.x[1]
        self.h[0], self.h[1] = c2p(xa,xb,self.w0*self.t)
        self.x[0].name = 'V'
        self.x[1].name = 'th'
        self.change_of_variable()


class Oscillator(HSS):
    def __init__(self):
        super().__init__(13)
    def setup(self):
        # States
        V,th = symbols('V th')
        self.x = [V, th]
        th0 = self.w0*self.t
        self.x0 = Matrix([1,0])

        # Inputs
        u = symbols('u')
        self.u = [u]
        self.u0 = [0 * self.t]

        # Parameters
        kv = symbols('kv')
        kth = symbols('kth')
        self.p = [kv, kth]
        self.p_value = [1, 5]

        # Differential equations
        #fx1 = self.w0*x2
        #fx2 = self.w0*(-2*zeta*x2-x1+beta*x1**3+cos(self.w0*self.t)+u)
        fV = (1-V)*kv
        fth = -th*kth
        self.f = [fV, fth]
        self.y = [V]

    def change_of_coordinates(self):
        self.h = [x for x in self.x]
        V = self.x[0]
        th = self.x[1]
        self.h[0], self.h[1] = p2c(V, th,-self.w0*self.t)
        self.x[0].name = 'xa'
        self.x[1].name = 'xb'
        self.change_of_variable()


class Oscillator_polar(HSS):
    def __init__(self):
        super().__init__(13)

    def setup(self):
        # States
        xa, xb = symbols('xa xb')
        self.x = [xa, xb]
        th0 = self.w0 * self.t
        self.x0 = Matrix([cos(self.w0*self.t), sin(self.w0*self.t)])

        # Inputs
        u = symbols('u')
        self.u = [u]
        self.u0 = [0 * self.t]

        # Parameters
        kv = symbols('kv')
        kth = symbols('kth')
        self.p = [kv, kth]
        self.p_value = [1, 5]

        # Algebraic
        V = (xa**2+xb**2)**0.5
        dV = kv*(1-V)
        th = atan2(xb,xa)
        dth = -kth*th+self.w0
        # Differential equations
        # fx1 = self.w0*x2
        # fx2 = self.w0*(-2*zeta*x2-x1+beta*x1**3+cos(self.w0*self.t)+u)
        fxa = dV/V*xa - dth*xb
        fxb = dV/V*xb + dth*xa
        self.f = [fxa, fxb]
        self.y = [V]

    def change_of_coordinates(self):
        self.h = [x for x in self.x]
        xa = self.x[0]
        xb = self.x[1]
        self.h[0], self.h[1] = c2p(xa, xb, -self.w0 * self.t)
        self.x[0].name = 'V'
        self.x[1].name = 'th'
        self.change_of_variable()


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
        ug = Vg*np.sqrt(2)*cos(self.w0*self.t+thgrid)

        xf = 0.052
        Cdc = 0.2e-3
        rf = 0.006
        lf = xf / self.w0
        cdc = Cdc * self.vdcb ** 2 / self.sb
        #cdc = cdc*1000

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
        vd0 = vref*cos(thgrid)
        vq0 = -vref*sin(thgrid)

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

    def change_of_coordinates(self):
        self.h = [x for x in self.x]
        self.h[1], self.h[2] = c2p(self.x[1], self.x[2], 0)
        self.x[1].name = 'V'
        self.x[2].name = 'th'

        self.h[3], self.h[4] = c2p(self.x[3], self.x[4],-self.w0*self.t)
        self.x[3].name = 'Isog'
        self.x[4].name = 'th_sog'
        self.change_of_variable()


#dvoc = dVOC_Statcom()
#dvoc.calc_modal_props()
#fig, axs = plt.subplots(2, sharex=True)
#dvoc_cov.pss.modal_props.plot_eigs(axs[0])

#duff = Oscillator_polar()
#duff.change_of_coordinates()
#duff.find_pss()
#fig, ax = plt.subplots()
#duff.pss.plot_states(ax)
#duff.calc_modal_props()
#fig2, ax2 = plt.subplots()
#duff.pss.modal_props.plot_eigs(ax2)
#sweep = ParametricSweep(duff)
#sweep.eigenloci(np.flip(np.linspace(0.1,5,30)), p_idx=0)
#sweep.eigenloci_plot()


dvoc = dVOC()
dvoc.find_pss()
dvoc.change_of_coordinates()
sweep = ParametricSweep(dvoc)
#sweep.eigenloci(np.flip(np.linspace(0.01,0.1,30)), p_idx=0)
sweep.eigenloci(np.linspace(0.01,0.1,30), p_idx=2)
sweep.eigenloci_plot()

#sweep.eigenloci(np.flip(np.linspace(0.001,0.1,30)), p_idx=2)

#dvoc.pss.modal_props.plot_eigs(axs[1])
#plt.show()
