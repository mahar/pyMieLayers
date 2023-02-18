
class Material(object): 
	"""
	Class Material defines the conductivities of the surface of each Layer.

	Parameters
	------

	"""

	def __init__(self,eps,mu,freqs,units="RADPERSEC"):
		'''
		@param eps: 
		@param mu
		@param ndarray freqs
		'''
		self.epsilon = eps*np.ones_like(freqs) + 0j
		self.mu = mu*np.ones_like(freqs) +0j
		self.impedance = np.sqrt(self.mu/self.epsilon)
		
		self.freq_range = freqs
		if units == "RADPERSEC" or units == None:
			self.freq_range = freqs
		elif units == "THz":
			self.freq_range = self.freq_range * 1e+12 * 2.0 * np.pi
		elif units == "GHz":
			self.freq_range = self.freq_range * 1e+9 * 2.0 * np.pi
		elif units == "m":
			self.freq_range = constants.c * 2.0 * np.pi / self.freq_range
		else:
			raise ValueError("Units are wrong...")

		self.RefrIndex = np.sqrt(self.epsilon*self.mu) + 0j
	
	def __add__(self,m1,m2):
		'''
		Adds two different epsilon
		'''
		if isinstance(m1, Material) and isinstance(m2,Material):
			return m1.epsilon + m2.epsilon 
		else:
			raise TypeError("")
	

class DispersiveMaterial(Material):
	def __init__(self,eps,mu,freqs,units="RADPERSEC"):
		'''
		@param eps: 
		@param mu
		@param ndarray freqs
		'''
		self.epsilon = eps+ 0j
		self.mu = mu +0j
		self.impedance = np.sqrt(mu/eps)
		
		self.freq_range = freqs
		if units == "RADPERSEC" or units == None:
			self.freq_range = freqs
		elif units == "THz":
			self.freq_range = self.freq_range * 1e+12 * 2.0 * np.pi
		elif units == "GHz":
			self.freq_range = self.freq_range * 1e+9 * 2.0 * np.pi
		elif units == "m":
			self.freq_range = constants.c * 2.0 * np.pi / self.freq_range
		else:
			raise ValueError("Units are wrong...")

		self.RefrIndex = np.sqrt(self.epsilon*self.mu) + 0j



class SurfaceMaterial(object):
	'''
	Class SurfaceMaterial defines the conductivities of the surface of each Layer.

	@TODO: 
	Incorporate Graphene as a subclass
	@param ndarray sigmaE 
	@param ndarray sigmaM
	@param ndarray freqs
	'''

	def __init__(self,sigmaE,sigmaM,freqs):
		self.sigmaE = sigmaE
		self.sigmaM = sigmaM 
		self.freq_range = freqs


class DrudeMaterial(Material):
	'''
	Class DrudeMaterial
	Inherits Class Material

	Adds a material with Lorentz Dispersion with multiple poles.

	epsilon(w) = 1-wp^2/(wp^2+i*w*Gamma)

	'''	
	def __init__(self,eps,mu,freq_range,params,units="RADPERSEC"):
		'''
		@param ndarray freq_range.
		@param (dict) params : params['GAMMA], params['PLASMA_FREQ']
		
		---  
		'''
		Material.__init__(self,eps,mu,freq_range,units=units)

		self.gamma = params['GAMMA']
		self.plasma_freq = params['PLASMA_FREQ']
	
		self.epsilon = eps - self.plasma_freq**2/(self.freq_range**2+1j*self.freq_range*self.gamma)

class LorentzMaterial(Material):
	'''
	Class LorentzMaterial
	Inherits Class Material

	Adds a material with Lorentz Dispersion with multiple poles.

	epsilon(w) = eps_inf + \sum ()/()

	'''
	def __init__(self,eps,mu,freq_range,poles=[],units="RADPERSEC"):
		'''
		@param ndarray freq_range.
		---  
		'''
		Material.__init__(self,eps,mu,freq_range,units=units)

		
		self.poles = poles

		if len(poles) > 0: 
			for pole in poles: 
				self.add_pole(pole['wT'],pole['wL'],pole['Gamma'])

	
	def _LorentzFunction(self,params):
		'''
		@param Dict pole: 
		'''
		return params['wL']
		

	def add_pole(self,wT,wL,Gamma):
		'''
		var poles should return a list of dictionaries
		'''
		_pole = {'wT': wT, 'wL': wL, 'Gamma': Gamma}
		self.poles.append(_pole)
		self._calculate(_pole)

	def _calculate(self,pole,newPole=True):
		'''
		Calculate the new dielectric function after the addition of a pole
		'''
		self.epsilon = self.epsilon + self._LorentzFunction(_pole)
		self.RefrIndex = np.sqrt(self.epsilon*self.mu)
		
class GainMaterial(Material):
	'''
	Class GainMaterial
	Inherits Class Material

	Adds a material with Gain. Based On Capolino's paper Eq. (14)

	epsilon(w) = eps_inf + \sum ()/()

	'''
	def __init__(self,eps,mu,freq_range,params=dict(),units="RADPERSEC"):
		'''
		@param ndarray freq_range.
		@param params: dict of 8 parameters
		R6G_Params = {'TAU_21': 3.99e-9, 'TAU_10' : 100e-15, 'TAU_32' : 100e-15, 
              'N0' : 6e24,
              'Gpump' : 1.5e-9, 
              'Omega' : 526e12*2.0*np.pi , 'DOmega':2.0*np.pi*27.7e12, 
              'Coupling' : 6.55e9} 
		---  
		'''
		Material.__init__(self,eps,mu,freq_range,units=units)

		#params 4-level model:
		DN = 0.0;
		self._tau21 = 0;
		self._tau10 = 0;
		self._tau32 = 0;
		self._N0 = 0.0;
		self._Gpump = 0;
		self._DOmega = 0.0
		self._Omega = 0.0
		self._Coupling = 0.0

		
		if 'TAU_21' in params.keys(): self._tau21 = params['TAU_21']
		if 'TAU_10' in  params.keys(): self._tau10 = params['TAU_10']
		if 'TAU_32' in params.keys(): self._tau32 = params['TAU_32']
		if 'N0' in params.keys(): self._N0 = params['N0']
		if 'Gpump' in params.keys(): self._Gpump = params['Gpump']
		if 'Omega' in params.keys(): self._Omega = params['Omega']
		if 'DOmega' in params.keys(): self._DOmega = params['DOmega']
		if 'Coupling' in params.keys(): self._Coupling = params['Coupling']
		

		DN = (self._tau21-self._tau10)*self._Gpump*self._N0/(1.0+(self._tau32+self._tau10+self._tau21)*self._Gpump)
		term = self._Coupling/(self.freq_range**2+1j*self.freq_range*self._DOmega-self._Omega**2)

		self.epsilon = eps + term*DN/Eps0
	
			
	
	def calculate(self, parameter_list):
		"""
		docstring
		"""
		pass

	