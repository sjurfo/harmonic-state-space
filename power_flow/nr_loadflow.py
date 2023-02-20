#
# Load flow solution with Newton-Raphson - line oriented approach
#
# Developed by: Sjur Føyen, Dept. of Electric Power Engineering, NTNU
#
#       September 2020
#
import numpy as np
from power_flow.read_data_with_trafos import read_data, Bus, Transformer
from typing import List


def admittance_matrix(bd):
    """
    Function is for the moment deprecated, as
    the NR solver adopts the line-oriented approach.
    :param bd:
    :return:
    """
    Nbus = len(bd)
    Ybus = 1j*np.zeros((Nbus,Nbus))
    for i in bd:
        selfY = 1j*i.comp
        for bus,line in i.connectors.items():
            Ybus[i.num,bus.num] = -line.y
            selfY += line.y
        Ybus[i.num,i.num] = selfY # set self-admittance
    return Ybus


def printsys(bd:List[Bus]):
    str = "{:^6}|{:^6}|{:^8}|{:^8}|{:^8}|{:^8}|{:^8}|{:^8}|{:^8}".format("#","Type","Pspec","Qspec",
                                                                         "Vspec","theta","V","P","Q")
    print(str)

    for i in bd:
        i.print()


def pv_limits(bd:List[Bus]):
    """
    Checks PV bus limits on reactive power
    :param bd: list of Bus objects
    :return: none
    """
    for i in bd:
        if i.type == 'PV':
            if i.Q > i.Qlim[1]:
                print('---------------------------------------------------------------------------')
                print('!!!!!!!!!!!!!!!!! Upper Q limit violated at bus {} !!!!!!!!!!!!!!!!!'.format(i.num))
                i.type = 'PQ'
                i.Q_spec = i.Qlim[1]

            if i.Q < i.Qlim[0]:
                print('---------------------------------------------------------------------------')
                print('!!!!!!!!!!!!!!!!! Lower Q limit violated at bus {}!!!!!!!!!!!!!!!!!'.format(i.num))
                i.type = 'PQ'
                i.Q_spec = i.Qlim[0]


def T(i:Bus,j:Bus, y):
    """
    Simplification of load flow equations
    :param i: frombus
    :param j: tobus
    :param y: line admittance
    :return: float
    """
    return np.real(y)*np.cos(i.angle-j.angle)+np.imag(y)*np.sin(i.angle-j.angle)


def U(i:Bus,j:Bus,y):
    """
    Simplification of load flow equations
    :param i: frombus
    :param j: tobus
    :param y: line admittance
    :return: float
    """
    return np.real(y) * np.sin(i.angle - j.angle) - np.imag(y) * np.cos(
        i.angle - j.angle)


def calculate_power(bd:List[Bus]):
    """
    Calculates and sets power for each Bus
    :param bd: List of Bus elements
    :return: none
    """
    for b_i in bd:
        b_i.P = 0
        b_i.Q = 0
        self_y = 1j * b_i.comp
        for bus, connector in b_i.connectors.items():
            #print(line.b_c2)
            yl, ysh = connector.get_admittance(b_i)
            self_y += ysh
            b_i.P -= b_i.V * bus.V * T(b_i, bus, yl)
            b_i.Q -= b_i.V * bus.V * U(b_i, bus, yl)
        b_i.P += b_i.V ** 2 * np.real(self_y)
        b_i.Q -= b_i.V ** 2 * np.imag(self_y)

    return None


def calc_jacobian(bd:List[Bus]):
    """
    "Meat" of the load flow solver. Computes the linearised power flow
    equations, popularly known as the Jacobian.
    The complexity of the code stems from the indices issue: bus indices
    are not the same as Jacobian indices.
    :param bd: List of Bus elements
    :return: Jacobian
    """
    j_ind1 = 0
    j_ind2 = 0
    for b_i in bd:
        if b_i.type == 'PQ':
            b_i.j_ind1 = j_ind1
            b_i.j_ind2 = j_ind2
            j_ind2 += 1
            j_ind1 += 1
        elif b_i.type == 'PV':
            b_i.j_ind1 = j_ind1
            j_ind1 += 1
    J1 = np.zeros((j_ind1,j_ind1))
    J2 = np.zeros((j_ind1,j_ind2))
    J3 = np.zeros((j_ind2,j_ind1))
    J4 = np.zeros((j_ind2,j_ind2))


    for b_i in bd:
        dP1 = 0
        dP2 = 0
        dQ1 = 0
        dQ2 = 0
        self_g = 0
        self_b = b_i.comp

        #print('{}, type = {}'.format(b_i.num,b_i.j_ind1))
        if b_i.j_ind1 is not None:  # Bus has angle in state variable - means upper part of Jacobian

            for bus, connector in b_i.connectors.items():
                yl, ysh = connector.get_admittance(b_i)
                self_g += np.real(ysh)
                dP1 += b_i.V * bus.V * U(b_i, bus, yl)
                dP2 -= bus.V * T(b_i, bus, yl)
                if bus.j_ind1 is not None:  # Bus has angle in x - this is J1
                    J1[b_i.j_ind1, bus.j_ind1] = -b_i.V * bus.V * U(b_i, bus, yl)
                if bus.j_ind2 is not None:  # Bus has voltage in x - this is J2
                    J2[b_i.j_ind1, bus.j_ind2] = -b_i.V * T(b_i, bus, yl)
            dP2 += 2 * b_i.V * self_g
            J1[b_i.j_ind1, b_i.j_ind1] = dP1
            if b_i.j_ind2 is not None:
                J2[b_i.j_ind1, b_i.j_ind2] = dP2

        if b_i.j_ind2 is not None:  # Bus has voltage in state variable - is lower part
            for bus, connector in b_i.connectors.items():
                yl, ysh = connector.get_admittance(b_i)
                self_b += np.imag(ysh)
                dQ1 -= b_i.V * bus.V * T(b_i, bus, yl)
                dQ2 -= bus.V * U(b_i, bus, yl)
                if bus.j_ind1 is not None:  # Bus has angle in x - this is J3
                    J3[b_i.j_ind2, bus.j_ind1] = b_i.V * bus.V * T(b_i, bus, yl)
                if bus.j_ind2 is not None:  # Bus has voltage in x - this is J4
                    J4[b_i.j_ind2, bus.j_ind2] = -b_i.V * U(b_i, bus, yl)
            #print(self_b)
            dQ2 -= 2 * b_i.V * self_b
            J3[b_i.j_ind2, b_i.j_ind1] = dQ1
            J4[b_i.j_ind2, b_i.j_ind2] = dQ2

    return np.vstack((np.hstack((J1,J2)),np.hstack((J3,J4))))  # Return the full Jacobian matrix


def newton_raphson_iteration(bd:List[Bus],jac):
    """
    A newton-raphson iteration.
    Sets up mismatch vector and solves (not by inversion) the linearised
    equation sets. Finally it updates the state variables.
    :param bd: List of Bus elements
    :param jac: Jacobian matrix
    :return: None
    """
    dP = []
    dQ = []
    a = 0
    for i in bd:
        if i.j_ind1 is not None:
            dP.append(i.P_spec-i.P)
            a += 1
        if i.j_ind2 is not None:
            dQ.append(i.Q_spec-i.Q)

    dS = np.asarray(dP+dQ)
    dx = np.linalg.solve(jac, dS)
    if max(abs(dx)) < 1e-7:
        return dS

    dx = dx
    for i in bd:
        if i.j_ind1 is not None:
            i.angle += dx[i.j_ind1]
        if i.j_ind2 is not None:
            i.V += dx[a+i.j_ind2]
    return np.array([False])

def newton_raphson(bd:List[Bus]):
    """
    The iterative scheme of NR load flow.
    :param bd: List of Bus elements
    :return: None
    """
    #calculate_power(bd)
    #print('---------------------------------------------------------------------------')
    #print('Initial conditions')
    #printsys(bd)
    #print('Initial Jacobian')
    #jac = calc_jacobian(bd)

    for i in range(5):
        pv_limits(bd)
        jac = calc_jacobian(bd)
        dS = newton_raphson_iteration(bd, jac)
        calculate_power(bd)
        #print('---------------------------------------------------------------------------')
        #print('Iteration {}'.format(i+1))
        #print('Jacobian:')
        #print(jac)
        #printsys(bd)
        if dS.any():
            print('Converged!')
            print('Largest mismatch = {}'.format(max(abs(dS))))
            break

    powerflows(bd)

    return None


def powerflows(bd:List[Bus]):
    #print('---------------------------------------------------------------------------')
    #print('Transmission line power flow')
    #print('----------------------------')
    #print(' Bus indices       P      Q    ')
    Ssum = 0
    for b_i in bd:
        for b_j, connector in b_i.connectors.items():
            yl, ysh = connector.get_admittance(b_i)
            current = b_i.V * np.exp(1j * b_i.angle)*ysh - b_j.V * np.exp(1j * b_j.angle) * yl
            b_i.powerflow[b_j] = b_i.V * np.exp(1j * b_i.angle) * np.conj(current)
            Ssum += b_i.powerflow[b_j]
        #b_i.printpf()
    #print('Total power losses = {}'.format(Ssum))


def main():
    bus_data, line_data = read_data('../data/datavoc.xlsx')
    # bus_data[1].angle = -0.05
    # bus_data[2].angle = -0.15
    # bus_data[2].V = 0.95
    # calculate_power(bus_data)
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)

    #calculate_power(bus_data)
    #print('---------------------------------------------------------------------------')
    #print('Initial conditions')
    #printsys(bus_data)
    jac = calc_jacobian(bus_data)
    print(jac)
    newton_raphson(bus_data)

    # Y = admittance_matrix(bus_data)
    # print(Y)

if __name__ == '__main__':
    #import cProfile
    #cProfile.run('main()')

    main()