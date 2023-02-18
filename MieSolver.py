
'''
@pyMieLayers
@ Developed by: Charalampos Mavidis
@ E-Mail: mavidis [at] iesl [dot] forth [dot] gr
@ Started: September 2020
@ Updated: 2023

@ Version: 2.0.0

@TO-DO
1) Materials frequencies should be calculated after build().
   Frequency range should be a property only of the main class

2) June 21/ Add field calculations.


#rsv361
'''
import numpy as np
from numpy.lib.arraysetops import isin
from scipy import special
from scipy.integrate import quad, romberg
from math import *
import scipy
from scipy.linalg import solve, norm,inv,pinv
from scipy import constants
import types

from .Layer import * 
from material import *


__version__ = '2.0.0'
__name__ = "pyMieLayers"

#constants
Z0 = constants.physical_constants['characteristic impedance of vacuum'][0]
Eps0 = constants.epsilon_0 

class MieSolver(object): 
    """MieSolver class"""
	
	
    def __init__(self,layers=[],freq_range=[],units="RADPERSEC"):
        '''
        
        PARAMETERS
        ----------
        `list` layers
        `list freq_range
        `str` units

        RETURNS
        ----------
        `MieSolver` object
        '''
		
        self.surfaceE_cond = 0.0 + 0j  # Surface Electric Conductivity
        self.surfaceM_cond = 0.0 + 0j  # Surface Electric Conductivity
        self.cond_term = 0+0j

        self.freqs = freq_range

        self._layers = layers # List of Layers
        self.total_matrixTE = 1.0+0j
        self.total_matrixTM = 1.0+0j

        self.host_layer = None

        self._built = False # Set to True if build() function has run successfully


        self.l1_kR = 0j
        self.Rmax = 0.0

        self.units = units
		
	
	
    def add_layer(self,Lay):
        '''
        Adds a Layer (class Layer) to the structure

        PARAMETERS
        -----------
        `Layer` layer
        '''

        # Check thickness and  Material 
        # Code here

        # Add
        if self._built==False:
            self._layers.append(Lay)

    def set_host(self,host_layer):
        '''
        set_host() sets the host layer

        PARAMETERS
        ----------
        `Layer` 
        '''
        self.host_layer = host_layer 
	
    def get_layers(self):
        '''
        get_layers

        RETURNS
        -------
        `list[Layer]`
        '''
        return self.Layers
	
    def set_freq_range(self,freqs):
        '''
        set_freq_range()

        PARAMETERS
        ----------
        `numpy.ndarray freqs
        
        '''
        self.freqs = freqs
		
        if self.units == "RADPERSEC" or self.units == None:
            self.freqs = freqs
        elif self.units == "THz":
            self.freqs = self.freqs * 1e+12 * 2.0 * np.pi
        elif self.units == "GHz":
            self.freqs = self.freqs * 1e+9 * 2.0 * np.pi
        elif self.units == "m":
            self.freqs = constants.c * 2.0 * np.pi / self.freqs
        else:
            raise ValueError("Units are wrong...")
        self.build(update=True)


    def build(self,update=False):
        '''
        Build the Transfer Matrix

        @param String calc: options={scattering efficiencies,coefficient of an order}

        '''
        if not self._built:
            print("Constructing Structure...")
            current_r = 0.0
            for layer_number,lay in enumerate(self._layers): 
                lay._layer_number = layer_number 
                lay.set_layer_bounds(current_r,current_r+lay.thickness)
                
            
                current_r = current_r + lay.thickness
            
                lay.set_R(current_r)
                
        
            self.Layers = np.array(self._layers)
            self._layers.clear()
            total_layers = len(self.Layers)
            self._built = True
            self.Rmax = self.Layers[len(self.Layers)-1].R
			
			
	# Observables
	
	#Scattering coeffs:
    def matrixTE(self,freq_index,layerCurrent,layerNext,order):
        '''
        Transfer Matrix Elements between current Layer and next Layer for TE polarization

        Each element of the matrix has shape (len(freqs),).
        '''
        
        z1 = layerCurrent.kR[freq_index]
        #print("K = ",layerNext.k)
        z2 = layerNext.k[freq_index]*layerCurrent.R
        
        #print(z1,z2)
        self.l1_kR = z1

        J1 = special.jv(order, z1)
        J2 = special.jv(order, z2)
        JD1 = special.jvp(order, z1)
        JD2 = special.jvp(order, z2)
        H1 = special.hankel1(order, z1)
        H2 = special.hankel1(order, z2)
        HD1 = special.h1vp(order, z1)
        HD2 = special.h1vp(order, z2)



        impedRatio = layerNext.impedance[freq_index]/layerCurrent.impedance[freq_index]
        termZ = 0


        impd2 = layerNext.impedance[freq_index]
        impd1 = layerCurrent.impedance[freq_index]
        sm = layerCurrent.sigmaM[freq_index]/Z0
        se = layerCurrent.sigmaE[freq_index]*Z0

        D2 = np.array([[JD2,HD2],[J2/impd2,H2/impd2]])
        D1 = np.array([[JD1,HD1],[J1/impd1,H1/impd1]])
        denom_ = H2*JD2-HD2*HD2
        D2inv = np.array([[H2,-impd2*HD2],[-J2,impd2*JD2]])/denom_

        #S = np.array([[1.0,-1j*sm],[1j*se,1.0]])
        s1 = np.array([[1.0,+1j*sm/2.0],[-1j*se/2.0,1.0]])
        s2 = np.array([[1.0,-1j*sm/2.0],[1j*se/2.0,1.0]])
        

        #p_matrix = np.matmul(S,D1)
        p_matrix = np.matmul(s2,D1)
        MatrixTE = np.matmul(inv(np.matmul(s1,D2)),p_matrix)
        return MatrixTE
	
	

    def matrixTM(self,freq_index,layerCurrent,layerNext,order):
        '''
        Transfer Matrix Elements between current Layer and next Layer for TM polarization
        '''
        z1 = layerCurrent.kR[freq_index]
        z2 = layerNext.k[freq_index]*layerCurrent.R
        self.l1_kR = z1

        J1 = special.jv(order, z1)
        J2 = special.jv(order, z2)
        JD1 = special.jvp(order, z1)
        JD2 = special.jvp(order, z2)
        H1 = special.hankel1(order, z1)
        H2 = special.hankel1(order, z2)
        HD1 = special.h1vp(order, z1)
        HD2 = special.h1vp(order, z2)

        impedRatio = layerNext.impedance[freq_index]/layerCurrent.impedance[freq_index]
        term_eZ = Z0#*layerNext.impedance[freq_index]
        term_mZ = layerCurrent.impedance[freq_index]/Z0
        magn = term_mZ*layerCurrent.sigmaM[freq_index]*1j

        term_e = 1j*term_eZ*layerCurrent.sigmaE[freq_index]

        impd2 = layerNext.impedance[freq_index]
        impd1 = layerCurrent.impedance[freq_index]
        sm = layerCurrent.sigmaM[freq_index]/Z0
        se = layerCurrent.sigmaE[freq_index]*Z0

        D2 = np.array([[J2,H2],[JD2/impd2,HD2/impd2]])
        D1 = np.array([[J1,H1],[JD1/impd1,HD1/impd1]])
        s2 = np.array([[1.0,1j*sm/2.0],[-1j*se/2.0,1.0]])
        s1 = np.array([[1.0,-1j*sm/2.0],[+1j*se/2.0,1.0]])
        p_matrix = np.matmul(s2,D1)
        MatrixTM = np.matmul(inv(np.matmul(s1,D2)),p_matrix)
        return MatrixTM

	######
	
	
	# Observables
    def get_ScattCoeff(self,order,polarization='all',inner=False):
        '''
        Function get_ScattCoeff 
        -----
        Calculates the scattering coefficient 
        ---
        Params
        -----
        @int order: order of the 
        @string polarization: 'all', 'TE' or 'TM'. Default: 'all
        Returns the scattering coefficient for one or more polarizations.
        @bool inner: If True the function will return the coefficient at 
        the innermost layer.
        '''

        #msg = "Calculating the total transfer matrix for m=%i..." % order
        #print(msg)
        coeffTE = np.ones_like(self.host_layer.k +0j)
        TE11 = np.ones_like(self.host_layer.k +0j)
        TE21 = np.ones_like(self.host_layer.k +0j)
        coeffTM = np.ones_like(self.host_layer.k +0j)

        

        self.total_matrixTE = np.array([[1+0j,0],[0j,1+0j]])
        self.total_matrixTM = np.array([[1+0j,0],[0j,1+0j]])

        S_matrixTE_Det = np.ones_like(self.host_layer.k +0j)
        S_matrixTE = np.array([[1+0j,0],[0j,1+0j]])

        if self._built: # check if build() function has run successfully
            for fi, kk in enumerate(self.host_layer.k):
                self.total_matrixTE = np.array([[1+0j,0j],[0j,1+0j]])
                self.total_matrixTM = np.array([[1+0j,0j],[0j,1+0j]])
                for layer_number,lay in enumerate(self.Layers): 

                    if layer_number == len(self.Layers)-1:

                        nextLayer = self.host_layer
                    else:
                        nextLayer = self.Layers[layer_number+1]
        
                    new_matrixTE = self.matrixTE(fi,lay,nextLayer,order)
                    new_matrixTM = self.matrixTM(fi,lay,nextLayer,order)

                    self.total_matrixTE = np.matmul(new_matrixTE,self.total_matrixTE)
                    self.total_matrixTM = np.matmul(new_matrixTM,self.total_matrixTM)
                    
                    TE11[fi] = self.total_matrixTE[0,0]
                    TE21[fi] = self.total_matrixTE[1,0]
                    
                    if inner: 
                        coeffTE[fi] = self.total_matrixTE[1,0]/self.total_matrixTE[0,0]
                        coeffTM[fi] = self.total_matrixTM[1,0]/self.total_matrixTM[0,0]
                    else:
                        coeffTE[fi] = self.total_matrixTE[1,0]/self.total_matrixTE[0,0]
                        coeffTM[fi] = self.total_matrixTM[1,0]/self.total_matrixTM[0,0]

                    S_matrixTE[0,0] = -self.total_matrixTE[1,0]/self.total_matrixTE[0,0]
                    S_matrixTE[1,1] = self.total_matrixTE[1,0]/self.total_matrixTE[0,0]
                    S_matrixTE[0,1] = 1.0/self.total_matrixTE[0,0]
                    S_matrixTE[1,0] = self.total_matrixTE[1,1]-self.total_matrixTE[1,0]/self.total_matrixTE[0,0] 
                    S_matrixTE_Det[fi] = np.linalg.det(S_matrixTE)+0j
            if polarization == 'TE':
                return coeffTE
            elif polarization == 'TM':
                return coeffTM
            elif polarization == 'TE11':
                return TE11,TE21
            elif polarization == 'S_TE':
                return S_matrixTE_Det

            else:
                return coeffTE, coeffTM
	
	#-------
	# Get coefficients for layer (i):
    def get_coeffs_at_layer(self,order,freq_index):
        """
        get_coeffs_at_layer() returns the coefficients at each layer
        
        PARAMETERS
        -----------
            order : int 
                multipole order
            frequency_index : int 
                index of the frequency array
        """
        if self._built: 
            
            # We start from the innermost layer
            bn_te,bn_tm = self.get_ScattCoeff(order,inner=True) 
            an_te,an_tm = self.get_ScattCoeff(order) 
            

            fi = freq_index
            field_te_n = np.array([[bn_te[fi]],[0j]],dtype=np.complex)
            field_tm_n = np.array([[bn_tm[fi]],[0j]],dtype=np.complex)

            coeff_in_te  = np.zeros(len(self.Layers)+1,dtype=np.complex)
            coeff_out_te = np.zeros(len(self.Layers)+1,dtype=np.complex)
            coeff_in_tm  = np.zeros(len(self.Layers)+1,dtype=np.complex)
            coeff_out_tm = np.zeros(len(self.Layers)+1,dtype=np.complex)

            #print(bn_te)

            coeff_in_te[0] = bn_te[fi]; coeff_out_te[0] = 0j
            coeff_in_tm[0] = bn_tm[fi]; coeff_out_tm[0] = 0j

            coeff_out_te[-1] = an_te[fi]; coeff_in_te[0] = 0.0
            coeff_out_tm[-1] = an_tm[fi]; coeff_in_tm[0] = 0.0

            for layer_number, lay in enumerate(self.Layers):
                if layer_number == len(self.Layers)-1:
                    nextLayer = self.host_layer
                    break
                else:
                    nextLayer = self.Layers[layer_number+1]
                
                new_matrixTE = self.matrixTE(fi,lay,nextLayer,order)
                new_matrixTM = self.matrixTM(fi,lay,nextLayer,order)
                
                field_te_n = np.matmul(new_matrixTE,field_te_n)
                field_tm_n = np.matmul(new_matrixTM,field_tm_n)
                

                coeff_in_te[layer_number+1] = field_te_n[0][0]; 
                coeff_out_te[layer_number+1] = field_te_n[1][0]
                coeff_in_tm[layer_number+1] = field_tm_n[0][0]; 
                coeff_out_tm[layer_number+1] = field_tm_n[1][0]

        return coeff_in_te,coeff_out_te, coeff_in_tm, coeff_out_tm


    def get_field_at_layer(self,order,freq_index,layer_number,orders=4):
        """
        get_field_at_layer returns the z-component of the field 
        for a scattering order.

        Parameters
        ----------
        order : int 
            Mie order. 
        freq_index : int 
            Frequency array index. 
        layer_number : int

        Returns
        -------
        field_TE,field_TM : `tuple` of `numpy.ndarray`

        
        """
        if layer_number==0: 
            bn_te,bn_tm = self.get_ScattCoeff(order,inner=True)
            layer = self.Layers[0]
            kk  = self.Layers[0].k[freq_index]
            
            bite = bn_te[freq_index]
            bitm = bn_tm[freq_index]
            fnte_in = lambda r,phi: bite * special.jv(order, kk*r) * np.cos(order*phi) 
            fntm_in = lambda r,phi: bitm * special.jv(order, kk*r) * np.cos(order*phi) 
            fieldZ_te = lambda r,phi: kk*fnte_in(r,phi)/layer.impedance[freq_index]
            fieldZ_tm = lambda r,phi: kk*fnte_in(r,phi)
            return fieldZ_te, fieldZ_tm
        
        if layer_number >= len(self.Layers): 
            # We are in host material
            layer = self.host_layer
            kk = layer.k[freq_index]
            layer_number = len(self.Layers)
            an_te,an_tm = self.get_ScattCoeff(order)
            bn_te,bn_tm = self.get_ScattCoeff(order)
            aite = an_te[freq_index]
            aitm = an_tm[freq_index]
            
            fieldZ_te = lambda r,phi: kk*(fnte_out(r,phi) + fnte_in(r,phi))
            fieldZ_tm = lambda r,phi: kk*(fnte_out(r,phi) + fnte_in(r,phi))

            fnte_out = lambda r,phi: aite * special.hankel1(order, kk*r) * np.cos(order*phi) 
            fntm_out = lambda r,phi: aitm * special.hankel1(order, kk*r) * np.cos(order*phi) 
            return fnte_out, fntm_out
        else:
            layer = self.Layers[layer_number]

        coeff_in_te,coeff_out_te, coeff_in_tm, coeff_out_tm = self.get_coeffs_at_layer(order,freq_index)
        c_in_te = coeff_in_te[layer_number]
        c_out_te = coeff_out_te[layer_number]
        c_in_tm = coeff_in_tm[layer_number]
        c_out_tm = coeff_out_tm[layer_number]

        kk = layer.k[freq_index]
        

        fnte_out = lambda r,phi: c_out_te * special.hankel1(order, kk*r)  * np.cos(order*phi) 
        fnte_in = lambda r,phi: c_in_te * special.jv(order, kk*r) * np.cos(order*phi) 

        fntm_out = lambda r,phi: c_out_tm * special.hankel1(order, kk*r) * np.cos(order*phi) 
        fntm_in = lambda r,phi: c_in_tm * special.jv(order, kk*r) * np.cos(order*phi) 

        

        fieldZ_te = lambda r,phi: kk*(fnte_out(r,phi) + fnte_in(r,phi))/layer.impedance[freq_index]
        fieldZ_tm = lambda r,phi: kk*(fntm_out(r,phi) + fntm_in(r,phi))
        return fieldZ_te, fieldZ_tm

    def get_field_at_layer_rphi(self,order,freq_index,layer_number,rho=True,orders=4,):
        """
        get_field_at_layer returns the rho and phi-components of the field 
        for a scattering order.

        Parameters
        ----------
        order : int 
            Mie order. 
        freq_index : int 
            Frequency array index. 
        layer_number : int

        Returns
        -------
        field_TE,field_TM : `tuple` of `numpy.ndarray`

        
        """
        
        if layer_number==0: 
            
                
            bn_te,bn_tm = self.get_ScattCoeff(order,inner=True)
            layer = self.Layers[0]
            kk  = self.Layers[0].k[freq_index]
            
            bite = bn_te[freq_index]
            bitm = bn_tm[freq_index]
            if rho==True:
                
                if order==0: 
                    fntm_in = lambda r,phi: 0.0
                    fnte_in = lambda r,phi: 0.0
                else:
                    fnte_in = lambda r,phi: -order*bite * special.jv(order, kk*r) * np.sin(order*phi) 
                    fntm_in = lambda r,phi: -order * bitm * special.jv(order, kk*r) * np.sin(order*phi)/r 
                fieldZ_te = lambda r,phi: kk*fnte_in(r,phi)/layer.impedance[freq_index]
                fieldZ_tm = lambda r,phi: kk*fnte_in(r,phi)
            return fieldZ_te, fieldZ_tm
        
        if layer_number >= len(self.Layers): 
            # We are in host material
            layer = self.host_layer
            kk = layer.k[freq_index]
            layer_number = len(self.Layers)
            an_te,an_tm = self.get_ScattCoeff(order)
            bn_te,bn_tm = self.get_ScattCoeff(order)
            aite = an_te[freq_index]
            aitm = an_tm[freq_index]
            
            fieldZ_te = lambda r,phi: kk*(fnte_out(r,phi) + fnte_in(r,phi))
            fieldZ_tm = lambda r,phi: kk*(fnte_out(r,phi) + fnte_in(r,phi))

            fnte_out = lambda r,phi: aite * special.hankel1(order, kk*r) * np.cos(order*phi) 
            fntm_out = lambda r,phi: aitm * special.hankel1(order, kk*r) * np.cos(order*phi) 
            return fnte_out, fntm_out
        else:
            layer = self.Layers[layer_number]

        coeff_in_te,coeff_out_te, coeff_in_tm, coeff_out_tm = self.get_coeffs_at_layer(order,freq_index)
        c_in_te = coeff_in_te[layer_number]
        c_out_te = coeff_out_te[layer_number]
        c_in_tm = coeff_in_tm[layer_number]
        c_out_tm = coeff_out_tm[layer_number]

        kk = layer.k[freq_index]
        

        fnte_out = lambda r,phi: c_out_te * special.hankel1(order, kk*r)  * np.cos(order*phi) 
        fnte_in = lambda r,phi: c_in_te * special.jv(order, kk*r) * np.cos(order*phi) 

        fntm_out = lambda r,phi: c_out_tm * special.hankel1(order, kk*r) * np.cos(order*phi) 
        fntm_in = lambda r,phi: c_in_tm * special.jv(order, kk*r) * np.cos(order*phi) 

        

        fieldZ_te = lambda r,phi: kk*(fnte_out(r,phi) + fnte_in(r,phi))/layer.impedance[freq_index]
        fieldZ_tm = lambda r,phi: kk*(fntm_out(r,phi) + fntm_in(r,phi))
        return fieldZ_te, fieldZ_tm
	
	#Scattering Amplitude 
    def get_scattAmplitude(self,theta,polarization,orders=50,mode=False):
        '''
        @param String polarization: "TE" or "TM"
        @param double theta: angle in radians
        @param Int orders: number of coefficients to be included (default: 50)
        @return <dict>: 

        '''

        if self._built:
            sc = 0.0+0j
            sc_orders = []
            

            for order in np.arange(0,orders):
                en = 2
                if ii==0: en = 1
                if polarization in ["TE","TM"]:
                    coeff = self.get_ScattCoeff(ii,polarization)
                    
                    sc = sc + en*coeff*np.cos(order*theta)
                    sc_orders.append(coeff*np.cos(order*theta))
            sc = (1-1j)*sc/np.sqrt(np.pi*self.host_layer.k)
            
            return {'amplitude':sc,'amplitude_ordered':np.array(sc_orders)}
			
	
	#Scattering efficiencies:
    def get_efficiencies(self,polarization,orders=50,mode=False):
        '''
        @param String polarization: "TE" or "TM"
        @param Int orders: number of coefficients to be included (default: 50)
        @return <dict>: 3 x ndarray 'Qext','Qsc','Qabs'

        '''

        if self._built:
            
            max_radius = self.Layers[len(self.Layers)-1].R
            #print("(in-code) max-radius = ", max_radius)
            khR = self.host_layer.k*max_radius
            #print('khR = ', khR)
            ext = 0.0
            sc = 0.0
            if mode>100:
                coeff = self.get_ScattCoeff(mode,polarization)
                Qsc = np.conjugate(coeff)*coeff/np.abs(khR)
                Qext = -np.real(coeff)/np.abs(khR)
                Qabs = Qext-Qsc
                Sc =  np.conjugate(coeff)*coeff

                #return {'Qext': Qext.real, 'Qsc': Qsc.real, 'Qabs': Qabs.real,
                    #	'Sc' : Sc}

            for ii in np.arange(0,orders):
                en = 2
                if ii==0: en = 1
                if polarization in ["TE","TM"]:
                    coeff = self.get_ScattCoeff(ii,polarization)
                    ext = ext + en*coeff.real
                    sc = sc + en*np.conjugate(coeff)*coeff
            Qsc = 2.0*sc.real/abs(khR)
            Qext = -2.0*ext.real/abs(khR)
            Qabs = Qext-Qsc
            Sc = sc.real
            return {'Qext': Qext, 'Qsc': Qsc, 'Qabs': Qabs,'Sc':Sc}
	
	# Effective Medium
    def get_effective(self,filling_ratio,polarization=None):
        '''
        get_effective_params
        @param (float) filling ratio: <1

        @return (dict) eff = {'EPS_PERP' : eps_perp,
                    'EPS_PAR' : eps_par,
                        'MU_PERP' : mu_perp,
                        'MU_PAR' : mu_par }
        '''
        if self._built:
            if filling_ratio>=1 or filling_ratio<=0:
                raise Exception("Filling Ratio should be between 0 and 1.")
            Rmax = self.Layers[len(self.Layers)-1].R
            #print("Rmax = ", Rmax)
            R_Np1 = Rmax/np.sqrt(filling_ratio)
            khR = self.host_layer.k*R_Np1
            khRmax = self.host_layer.k*Rmax
            #print('Rnp1 = ', R_Np1)
            

            J0 = special.jv(0, khR)
            H0 = special.hankel1(0, khR)
            JD0 = special.jvp(0, khR)
            HD0 = special.h1vp(0, khR)
            J1 = special.jv(1, khR)
            H1 = special.hankel1(1, khR)
            JD1 = special.jvp(1, khR)
            HD1 = special.h1vp(1, khR)

            a0_te = self.get_ScattCoeff(0,polarization='TE')
            a1_te = self.get_ScattCoeff(1,polarization='TE')
            a0_tm = self.get_ScattCoeff(0,polarization='TM')
            a1_tm = self.get_ScattCoeff(1,polarization='TM')

            term_0d_te = JD0 + HD0 * a0_te + 0j
            term_0_te = (J0 + H0 * a0_te) + 0j
            term_0d_tm = JD0 + HD0 * a0_tm + 0j
            term_0_tm = (J0 + H0 * a0_tm) + 0j

            term_1d_te = JD1 + HD1 * a1_te + 0j
            term_1_te = (J1 + H1 * a1_te) + 0j
            term_1d_tm = JD1 + HD0 * a1_tm + 0j
            term_1_tm = (J1 + H1 * a1_tm) + 0j

            eps_par = (-2.0*self.host_layer.epsilon/khR)*term_0d_tm/term_0_tm + 0j
            eps_perp = (self.host_layer.epsilon/khR)*term_1_te/term_1d_te + 0j
            mu_par = (-2.0*self.host_layer.mu/khR)*term_0d_te/term_0_te + 0j
            mu_perp = (self.host_layer.mu/khR)*term_1_tm/term_1d_tm + 0j

            # small params
            n1e1 = khRmax**2 - filling_ratio * a1_te*4j/np.pi
            d1e1 = khRmax**2 + filling_ratio * a1_te*4j/np.pi
            eps_perp_limit = self.host_layer.epsilon*n1e1/d1e1 + 0j

            n1m0 = khRmax**2 - filling_ratio * a0_tm*4j/np.pi
            d1m0 = khRmax**2 + filling_ratio * a0_tm*4j/np.pi
            mu_perp_limit = self.host_layer.mu*n1m0/d1m0 + 0j

            n2m0 = 1.0 - (filling_ratio/khRmax) * a0_tm*4j/np.pi
            eps_par_limit = self.host_layer.epsilon*n2m0 + 0j

            n2m1 = 1.0 - (filling_ratio/khRmax) * a1_tm*4j/np.pi
            mu_par_limit = self.host_layer.mu*n2m1 + 0j
            

            

            eff = {'EPS_PERP' : eps_perp,
                    'EPS_PAR' : eps_par,
                        'MU_PERP' : mu_perp,
                        'MU_PAR' : mu_par,
                        'EPS_PERP_LIMIT' : eps_perp_limit,
                        'EPS_PAR_LIMIT' : eps_par_limit,
                        'MU_PERP_LIMIT' : mu_perp_limit,
                        'MU_PAR_LIMIT' : mu_par_limit   
                    }

            return eff

	# Effective Medium
    def get_effective_higher(self,filling_ratio,polarization=None):
        '''
        get_effective_params
        @param (float) filling ratio: <1

        @return (dict) eff = {'EPS_PERP' : eps_perp,
                    'EPS_PAR' : eps_par,
                        'MU_PERP' : mu_perp,
                        'MU_PAR' : mu_par }
        '''
        if self._built:
            if filling_ratio>=1 or filling_ratio<=0:
                raise Exception("Filling Ratio should be between 0 and 1.")
            Rmax = self.Layers[len(self.Layers)-1].R
            R_Np1 = Rmax/np.sqrt(filling_ratio)
            khR = self.host_layer.k*R_Np1
            #print('Rnp1 = ', R_Np1)
            

            J2 = special.jv(2, khR)
            H2 = special.hankel1(2, khR)
            JD2 = special.jvp(2, khR)
            HD2 = special.h1vp(2, khR)
            J3 = special.jv(3, khR)
            H3 = special.hankel1(3, khR)
            JD3 = special.jvp(3, khR)
            HD3 = special.h1vp(3, khR)

            a2_te = self.get_ScattCoeff(2,polarization='TE')
            a3_te = self.get_ScattCoeff(3,polarization='TE')
            a2_tm = self.get_ScattCoeff(2,polarization='TM')
            a3_tm = self.get_ScattCoeff(3,polarization='TM')

            term_2d_te = JD2 + HD2 * a2_te + 0j
            term_2_te = (J2 + H2 * a2_te) + 0j
            term_2d_tm = JD2 + HD2 * a2_tm + 0j
            term_2_tm = (J2 + H2 * a2_tm) + 0j

            term_3d_te = JD3 + HD3 * a3_te + 0j
            term_3_te = (J3 + H1 * a3_te) + 0j
            term_3d_tm = JD3 + HD3 * a3_tm + 0j
            term_3_tm = (J3 + H3 * a3_tm) + 0j

            #11. Get Initial estimation of eps, mu 
            eff_start = self.get_effective(filling_ratio)
            eps_par = eff_start['EPS_PAR']  + 0j
            eps_perp = eff_start['EPS_PERP']  + 0j
            mu_perp = eff_start['MU_PERP']  + 0j
            mu_par = eff_start['MU_PAR']  + 0j

            # m=2
            k0R = self.Layers[0].k0 * R_Np1 
            neff_TE = (8.0/k0R) * (term_2_te/term_2d_te) + 0j 

            imped_eff = (neff_TE/3.0) * (term_3d_te/term_3_te) + 0j

            eff_perp = neff_TE/imped_TE 
            mu_par = neff_TE*imped_TE 
            xm = 0
            xeff = neff_TE*k0R 
            while np.abs(xeff-xm)>1e-6:
                imped_eff = (neff_TE/3.0) * (term_3d_te/term_3_te) + 0j
                eff_perp = neff_TE/imped_TE 
                mu_par = neff_TE*imped_TE
                xeff = neff_TE*k0R 


            eff = {'EPS_PERP' : eps_perp,
                    'EPS_PAR' : eps_par,
                        'MU_PERP' : mu_perp,
                        'MU_PAR' : mu_par
            }

            return eff
	



		


