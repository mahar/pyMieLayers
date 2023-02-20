'''
Calculate the conductivity of Graphene
using Kubo's formula in the context
of the Random Phase Approximation
'''
import numpy as np
from scipy import constants as const
from scipy import integrate
import scipy as sp

# Constants
kb = const.k  # in J/K
hbar = const.hbar  # in J*sec
e_el = const.e  # Coulomb
exp = lambda x: np.exp(x)

conductivity_unit = e_el ** 2 / (4. * hbar)


class GrapheneConductivity(object):
    '''GrapheneConductivity class'''

    def __init__(self, freq, Ef, tau, T, use_intergrand=True,photoexcited=False):
        '''
        Class constructor:
        @param frequency: angular frequency (1/s)
        @param Ef: fermi energy in eV
        @param tau: relaxation time in sec
        @param T: temperature in K
        @use_intergrand: Uses the intergral to compute the second term
		@photoexcited: Using Semi Fermi levels if True. Default: False
        @returns array, array
        '''
        self._freq = freq
        self._EfJ = e_el * Ef
        self._Ef = Ef
        self._tau = tau
        self._temperature = T

        self.use_intergrand = use_intergrand 

        self._intERband = 0.0 + 0j
        self._intRAband = 0.0 + 0j

        self._intrabandConstant = 0.0

        self._calculated = False
        if photoexcited:
            self.calculate_photoexcited()
        else:
            self.calculate()

    def calculate(self, frequency=0):
        '''
        @param frequency: angular frequency (1/s)
        @return None
        '''
        if frequency > 0:
            self.set_frequency(frequency)

        if self._temperature < 5.0:
            intRAband = (e_el * e_el * self._EfJ / (np.pi * hbar * hbar)) * 1j / (self._freq + 1j / self._tau)
            intERband = conductivity_unit * 1j * sp.log(abs(hbar * self._freq - 2 * self._EfJ)) / np.pi

            # Step Function
            if hbar * self._freq > 2 * self._EfJ:
                intERband += conductivity_unit

        else:  # Higher temperatures

            const_ = 2 * e_el ** 2 * kb * self._temperature / np.pi / hbar ** 2
            fr = 1j / (self._freq + 1j / self._tau)
            # comsol 
            # (2*ee^2*kB*T/(pi*hbar^2))*1/(gam-i*om)*log(2*cosh(eF/(2*kB*T)))
            self._intrabandConstant = 1j* const_  * sp.log(2 * sp.cosh(self._EfJ / 2. / kb / self._temperature))
            intRAband = const_ * fr * sp.log(2 * sp.cosh(self._EfJ / 2. / kb / self._temperature))
            #print("prefactor =",const_ *  sp.log(2 * sp.cosh(self._EfJ / 2. / kb / self._temperature)))
            #print("Ef = ", self._EfJ/e_el)
            ##print("intraband = ", intRAband/fr)
            #print("prefactor =",sp.log(2 * sp.cosh(self._EfJ / 2. / kb / self._temperature))/sp.log(2.0))


            if self.use_intergrand:
                #print("use intergrand method...")
                integr = integrate.quad(self.integrand, 0, self._freq -self._freq / 1e9, epsrel=1e-22)[0]
                intERband = (self.H(self._freq / 2) + 4j * self._freq * integr / np.pi) * conductivity_unit
            else:
                #print("i am here ....")
                term1 = (hbar * self._freq - 2 * self._EfJ)/(2*kb*self._temperature)
                t3 = (hbar*self._freq+2*self._EfJ)**2
                t4 = (hbar*self._freq-2*self._EfJ)**2+4*(kb*self._temperature)**2
                term2 = t3/t4 + 0j
                intERband = conductivity_unit*(0.5+ sp.arctan(term1)/ np.pi -1j*np.log(term2)/(2*np.pi))
            # print self._freq,self._Ef
            # sINTER = conductivityUnit*(H(freq/2) + 4j*freq.*ii/pi);

        self._intERband = intERband
        self._intRAband = intRAband


        if abs(self._intRAband) > 0 or abs(self._intERband) > 0:
            self._calculated = True

    def H(self, omega):
        '''
        '''
        a = sp.sinh(hbar * omega / kb / self._temperature)
        b = sp.cosh(self._EfJ / kb / self._temperature) + sp.cosh(hbar * omega / kb / self._temperature)
        return a / b

    def calculate_photoexcited(self, frequency=0):

        if frequency > 0:
            self.set_frequency(frequency)

        if self._temperature < 5.0:
            intRAband = (e_el * e_el * self._EfJ / (np.pi * hbar * hbar)) * 1j / (self._freq + 1j / self._tau)
            intERband = conductivity_unit * 1j * sp.log(abs(hbar * self._freq - 2 * self._EfJ)) / np.pi

            # Step Function
            if hbar * self._freq > 2 * self._EfJ:
                intERband += conductivity_unit

        else:  # Higher temperatures

            const_ = e_el ** 2 / hbar
            Efk = self._EfJ / (kb * self._temperature)

            fr1 = 1.0 / (1.0 - 1j * self._freq * self._tau)

            term1 = 8 * kb * self._temperature * self._tau / (4.0 * np.pi * hbar) * np.log(1.0 + np.exp(Efk))
            term2 = sp.tanh((hbar * self._freq - self._EfJ) / (4.0 * kb * self._temperature))

            intRAband = const_ * (term1 * fr1 + term2 / 4.0)

            integr = integrate.quad(self.integrand, 0, self._freq - self._freq / 1e9, epsrel=1e-22)[0]
            intERband = (4j * self._freq * integr / np.pi) * conductivity_unit
            # print self._freq,self._Ef
            # sINTER = conductivityUnit*(H(freq/2) + 4j*freq.*ii/pi);

        self._intERband = intERband
        self._intRAband = intRAband

        if abs(self._intRAband) > 0 or abs(self._intERband) > 0:
            self._calculated = True

    def integrand(self, w):
        term1 = self.H(self._freq * 0.5)
        term2 = self.H(w)
        return (term2 - term1) / (self._freq ** 2 - 4.0 * w ** 2)

    def set_frequency(self,omega):
        '''
        Set frequency in rad/s
        '''
        self._freq = omega

    def get_intraband(self):
        return self._intRAband

    def get_interband(self):
        return self._intERband
    
    def get_intrabandConstant(self):
        return self._intrabandConstant

    def get_conductivity(self):
        return self._intERband + self._intRAband

    def get_impedance(self):
        if self._calculated:
            conducti = self.get_conductivity()
            return 1.0 / conducti
        return 0
