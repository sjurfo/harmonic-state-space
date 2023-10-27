import numpy as np
from sympy import symbols, cos, sin, Matrix, atan2
from hsslib.hss import HSS


class ProportionalResonantCCVSC(HSS):
    def setup(self):
        # States
        ua, ub, = symbols('ua ub')  # SOGI
        xa, xb = symbols('xa xb')  # Resonator
        ia = symbols('ia')
        self.x = [ia, ua, ub, xa, xb]
        # Initial guess not important here!
        self.x0 = Matrix([1,1,1,1,1])

        # Inputs
        up = symbols('up')
        self.u = [up]
        ug = cos(self.w0 * self.t) + up

        # Parameters
        kp_cc, ksog = symbols('kp_cc ksog')
        self.p = [kp_cc, ksog]
        self.p_value = [1, np.sqrt(2)]

        id_ref = 1;    iq_ref = 0
        lf = 0.04;     rf = 0.005
        a_cancel = (lf / rf + rf / lf) * 0.5
        ki_cc = 2*kp_cc*a_cancel

        # Algebraic equations
        th = atan2(ub, ua)
        dia = (id_ref * cos(th) - iq_ref * sin(th) - ia)
        uc = xa * ki_cc + kp_cc * dia

        # Differential equations
        fua = (ksog * (ug - ua) * self.w0 - ub * self.w0)
        fub = ua * self.w0
        fxa = self.w0 * (dia - xb)
        fxb = xa * self.w0
        fia = (self.w0 / lf * (uc - ug - ia * rf))
        self.f = [fia, fua, fub, fxa, fxb]

        # Outputs
        self.y = [-ia]


from hsslib.htf import HTF
cc_vsc = ProportionalResonantCCVSC(N=13)
cc_vsc.find_pss()
cc_vsc.toep_BCD()

Y = HTF(cc_vsc, 5*np.logspace(0,3,250))
Y.plot(Nplot=3,xscale='log')
Ym = Y.get_mid_col()
axs = Ym.plot(xscale='log')

#fig.text(0.5, 0.04, r'$f$ $[Hz]$', ha='center', size='12')
#fig.text(0.04, 0.5, r'$|H|$ $[pu]$', va='center', rotation='vertical', size='12')
#axs[1].set_xlabel(r'$f$ $[Hz]$')
#axs[0].set_ylabel(r'$|H|$ $[pu]$')
#axs[1].set_ylabel(r'$\angle{H}$ $[rad]$')

#plt.savefig('y_prcc_midcol', bbox_inches="tight", pad_inches=0.3, transparent=True, format='png', dpi=600)