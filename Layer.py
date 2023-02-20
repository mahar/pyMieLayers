

class Layer(object):
	'''
	class  Layer

	Each Layer has each own thickness, Material and SurfaceMaterial (optional)

	'''

	def __init__(self,thickness,material,surface_material=None,name=""):
		self.thickness = thickness
		self.epsilon = material.epsilon
		self.freq_range = material.freq_range
		self.mu = material.mu
		self.RefrIndex = np.sqrt(material.epsilon*material.mu)
		self.sigmaE = np.zeros_like(material.freq_range)+0j
		self.sigmaM = np.zeros_like(material.freq_range)+0j
		# Check if surface_material is an instance of the class SurfaceMaterial
		if isinstance(surface_material,SurfaceMaterial):
			self.sigmaE = surface_material.sigmaE
			self.sigmaM = surface_material.sigmaM

		self.name = name
		self._layer_number = -1 # Assign a number based on the order

		self.kR = 0j
		self.R = thickness
		self.k = self.RefrIndex*self.freq_range/constants.c
		self.k0 = self.freq_range/constants.c

		self.impedance = np.sqrt(self.mu/self.epsilon)
		
		# get layer boundaries
		self.r_start = 0.0
		self.r_end = 0.0
	
	def __str__(self):
		return self.name
	
	def __repr__(self):
		return str(self.name)

	

	def set_R(self,R):
		# Set the distance from the center of the cylinder to the edge
		self.R = R
		self.k = self.RefrIndex*self.freq_range/constants.c
		self.kR = self.RefrIndex*self.freq_range*self.R/constants.c
	
	def set_layer_bounds(self,rstart,rend):
		self.r_start = rstart
		self.r_end = rend
	