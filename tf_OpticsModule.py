# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 22:30:30 2018

@author: deepLearning505
"""

import tensorflow as tf
import numpy as np
import math

def tf_fft_shift_2d(self):

    B, M, N = self.shape
    if np.mod(M.value, 2) == 0:
        M_half = M.value / 2.0
    else:
        M_half = np.floor(M.value / 2.0) + 1.0
    if np.mod(N.value, 2) == 0:
        N_half = N.value / 2.0
    else:
        N_half = np.floor(N.value / 2.0) + 1.0

    img_1 = tf.slice(self, np.int32([0, 0, 0]), np.int32([B, M_half, N_half]))
    img_2 = tf.slice(self, np.int32([0, 0, N_half]), np.int32([B, M_half, N - N_half]))
    img_3 = tf.slice(self, np.int32([0, M_half, 0]), np.int32([B, M - M_half, N_half]))
    img_4 = tf.slice(self, np.int32([0, M_half, N_half]), np.int32([B, M - M_half, N - N_half]))

    return tf.concat([tf.concat([img_4, img_3], 2), tf.concat([img_2, img_1], 2)], 1)

def tf_ifft_shift_2d(self):
    
    B, M, N = self.shape
    if np.mod(M.value, 2) == 0:
        M_half = M.value / 2.0
    else:
        M_half = np.floor(M.value / 2.0)
    if np.mod(N.value, 2) == 0:
        N_half = N.value / 2.0
    else:
        N_half = np.floor(N.value / 2.0)

    img_1 = tf.slice(self, np.int32([0, 0, 0]), np.int32([B, M_half, N_half]))
    img_2 = tf.slice(self, np.int32([0, 0, N_half]), np.int32([B, M_half, N - N_half]))
    img_3 = tf.slice(self, np.int32([0, M_half, 0]), np.int32([B, M - M_half, N_half]))
    img_4 = tf.slice(self, np.int32([0, M_half, N_half]), np.int32([B, M - M_half, N - N_half]))

    return tf.concat([tf.concat([img_4, img_3], 2), tf.concat([img_2, img_1], 2)], 1)

def tf_flattop(f0,M,N):
    
    flattop_filter = np.zeros((M,N)) # TO BE CONTINUED...
    
    return flattop_filter

def tf_udft2(self):
    
    M,N = self.shape[1],self.shape[2]
    # self = tf.convert_to_tensor(self, dtype=tf.complex64)
    out = tf.fft2d(self)/math.sqrt(M.value*N.value)
    return out

def tf_uidft2(self):
    
    M,N = self.shape[1],self.shape[2]
    # self = tf.convert_to_tensor(self, dtype=tf.complex64)
    out = tf.ifft2d(self)*math.sqrt(M.value*N.value)
    return out

def tf_sdft2(self):
    
    B,M,N = self.shape #M, N = self.get_shape().as_list()
    x = tf.constant(np.arange(M.value),shape=[M.value,1],dtype=tf.float32)
    y = tf.constant(np.arange(N.value),shape=[1,N.value],dtype=tf.float32)
    xphase = tf.matmul(np.pi*(M.value-1)/M.value*x,tf.ones((1,N.value),dtype=tf.float32))
    yphase = tf.matmul(tf.ones((M.value,1),dtype=tf.float32),np.pi*(N.value-1)/N.value*y)
    xyphase = xphase+yphase
#    x = np.arange(M.value)
#    y = np.arange(N.value)
#    xyphase = np.outer(np.pi*(M.value-1)/M.value*x,np.ones(N.value))+np.outer(np.ones(M.value),np.pi*(N.value-1)/N.value*y)
    exy = tf.complex(tf.cos(xyphase),tf.sin(xyphase))
    exy = tf.cast(exy,dtype=tf.complex64)
#    exy = tf.tile([exy],[B.value,1,1])
#    exy = np.outer(np.exp(1j*np.pi*(M.value-1)/M.value*x),(np.exp(1j*np.pi*(N.value-1)/N.value*y)))
#    phaseterm = np.exp(-1j*math.pi*((M.value-1)**2)/2/M.value)*np.exp(-1j*math.pi*((N.value-1)**2)/2/N.value)*exy
    phaseterm = tf.complex(tf.cos(-math.pi*((M.value-1)**2)/2/M.value-math.pi*((N.value-1)**2)/2/N.value),\
                           tf.sin(-math.pi*((M.value-1)**2)/2/M.value-math.pi*((N.value-1)**2)/2/N.value))
    phaseterm = tf.cast(phaseterm,dtype=tf.complex64)
#    phaseterm = tf.tile([phaseterm],[B.value,1,1])
    phaseterm = tf.multiply(phaseterm,exy)
    output = tf.multiply(phaseterm,tf_udft2(tf.multiply(self,exy)))
    return output

def tf_sidft2(self):
    
    B,M,N = self.shape
    x = tf.constant(np.arange(M.value),shape=[M.value,1],dtype=tf.float32)
    y = tf.constant(np.arange(N.value),shape=[1,N.value],dtype=tf.float32)
    xphase = tf.matmul(-np.pi*(M.value-1)/M.value*x,tf.ones((1,N.value),dtype=tf.float32))
    yphase = tf.matmul(tf.ones((M.value,1),dtype=tf.float32),-np.pi*(N.value-1)/N.value*y)
    xyphase = xphase+yphase
    exy = tf.complex(tf.cos(xyphase),tf.sin(xyphase))
    exy = tf.cast(exy,dtype=tf.complex64)
#    exy = tf.tile([exy],[B.value,1,1])
#    exy = np.outer(np.exp(-1j*np.pi*(M.value-1)/M.value*x),np.exp(-1j*np.pi*(N.value-1)/N.value*y))
#    phaseterm = np.exp(1j*np.pi*((M.value-1)**2)/2/M.value)*np.exp(1j*np.pi*((N.value-1)**2)/2/N.value)*exy
    phaseterm = tf.complex(tf.cos(np.pi*((M.value-1)**2)/2/M.value+np.pi*((N.value-1)**2)/2/N.value),\
                           tf.sin(np.pi*((M.value-1)**2)/2/M.value+np.pi*((N.value-1)**2)/2/N.value))
    phaseterm = tf.cast(phaseterm,dtype=tf.complex64)
#    phaseterm = tf.tile([phaseterm],[B.value,1,1])
    phaseterm = tf.multiply(phaseterm,exy)
    output = tf.multiply(phaseterm,tf_uidft2(tf.multiply(self,exy)))
    return output

def tf_FSPAS(self,wlength,z,dx,dy,ridx,*theta0):
    
    """
    Angular Spectrum Propagation of Coherent Wave Fields
    with optional filtering
    
    INPUTS : 
        U, wave-field in space domain
        wlenght : wavelength of the optical wave
        z : distance of propagation
        dx,dy : sampling intervals in space
        M,N : Size of simulation window
        theta0 : Optional BAndwidth Limitation in DEGREES 
            (if no filtering is desired, only EVANESCENT WAVE IS FILTERED)
    
    OUTPUT : 
        output, propagated wave-field in space domain
    
    """
    
    wlengtheff = wlength/ridx
    B,M,N = self.shape
    dfx = 1/dx/M.value
    dfy = 1/dy/N.value
    fx = tf.constant((np.arange(M.value)-(M.value-1)/2)*dfx,shape=[M.value,1],dtype=tf.float32)
    fy = tf.constant((np.arange(N.value)-(N.value-1)/2)*dfy,shape=[1,N.value],dtype=tf.float32)
    fx2 = tf.matmul(fx**2,tf.ones((1,N.value),dtype=tf.float32))
    fy2 = tf.matmul(tf.ones((M.value,1),dtype=tf.float32),fy**2)
#    fx = (np.arange(M.value)-(M.value-1)/2)*dfx
#    fy = (np.arange(N.value)-(N.value-1)/2)*dfy
#    fx2 = np.outer((fx)**2,np.ones(N.value))
#    fy2 = np.outer(np.ones(M.value),(fy)**2)
    if theta0:#BANDLIMIT OF THE FREE-SPACE PROPAGATION
        f0 = np.sin(np.deg2rad(theta0))/wlengtheff
        Q = tf.to_float(((fx2+fy2)<=(f0**2)))
    else:
        Q = tf.to_float(((fx2+fy2)<=(1/wlengtheff**2)))
    
#    Q = Q.astype(int)
    W = Q*(fx2+fy2)*(wlengtheff**2)
    Hphase = 2*np.pi/wlengtheff*z*(tf.ones((M.value,N.value))-W)**(0.5)
    HFSP = tf.complex(Q*tf.cos(Hphase),Q*tf.sin(Hphase))     
#    HFSP = np.exp(1j*2*np.pi/wlengtheff*z*(np.ones((M.value,N.value))-W)**(0.5))*Q
#    HFSP = tf.convert_to_tensor(HFSP, dtype=tf.complex64)
#    HFSP = tf.tile([HFSP],[B.value,1,1])        
    output = tf_sidft2(tf.multiply(HFSP,tf_sdft2(self)))
    output = tf.slice(output, np.int32([0, 0, 0]), np.int32([B, M.value, N.value]))
    return output

def tf_FSPAS_FFT(self,wlength,z,dx,dy,ridx,*theta0):
    
    """
    Angular Spectrum Propagation of Coherent Wave Fields
    with optional filtering
    
    INPUTS : 
        U, wave-field in space domain
        wlenght : wavelength of the optical wave
        z : distance of propagation
        dx,dy : sampling intervals in space
        M,N : Size of simulation window
        theta0 : Optional BAndwidth Limitation in DEGREES 
            (if no filtering is desired, only EVANESCENT WAVE IS FILTERED)
    
    OUTPUT : 
        output, propagated wave-field in space domain
    
    """
    
    wlengtheff = wlength/ridx
    B,M,N = self.shape
    dfx = 1/dx/M.value
    dfy = 1/dy/N.value
    fx = tf.constant((np.arange(M.value)-(M.value)/2)*dfx,shape=[M.value,1],dtype=tf.float32)
    fy = tf.constant((np.arange(N.value)-(N.value)/2)*dfy,shape=[1,N.value],dtype=tf.float32)
    fx2 = tf.matmul(fx**2,tf.ones((1,N.value),dtype=tf.float32))
    fy2 = tf.matmul(tf.ones((M.value,1),dtype=tf.float32),fy**2)
    if theta0:#BANDLIMIT OF THE FREE-SPACE PROPAGATION
        f0 = np.sin(np.deg2rad(theta0))/wlengtheff
        Q = tf.to_float(((fx2+fy2)<=(f0**2)))
    else:
        Q = tf.to_float(((fx2+fy2)<=(1/wlengtheff**2)))
    
    W = Q*(fx2+fy2)*(wlengtheff**2)
    Hphase = 2*np.pi/wlengtheff*z*(tf.ones((M.value,N.value))-W)**(0.5)
    HFSP = tf.complex(Q*tf.cos(Hphase),Q*tf.sin(Hphase))
    ASpectrum = tf.fft2d(self)
    ASpectrum = tf_fft_shift_2d(ASpectrum)    
    ASpectrum_z = tf_ifft_shift_2d(tf.multiply(HFSP,ASpectrum))
    output = tf.ifft2d(ASpectrum_z)
    output = tf.slice(output, np.int32([0, 0, 0]), np.int32([B, M.value, N.value]))
    return output

def paraxlens(wlength,focal_length,center,fshift,diameter,dx,dy,M,N):
    
    """
    Create function of a ideal paraxial lens
    INPUTS : 
        wlength -> wavelength of the optical signal
        f -> 2-by-1 list contains -> focal length in x and focal length in y
        center -> 2-by-1 list contains -> central shift of whole lens func. in x and y
        fshift -> lateral off-axis shift on the focal plane
        dia -> diameter of lens aperture
        dx,dy -> sampling intervals in space
        M,N -> size of simulation window
        
    OUTPUTS:
        lensfunc -> numerical lensfunction defined in space
        
    """
    x = (np.arange(M)-(M-1)/2)*dx
    y = (np.arange(N)-(N-1)/2)*dy
    x0,y0 = center[0], center[1]
    xf,yf = fshift[0], fshift[1]
    Lxy2 = (np.outer((x-x0)**2,np.ones(N))+np.outer(np.ones(M),(y-y0)**2))
    D = Lxy2<=(diameter**2)
    sq = np.outer((x-xf)**2,np.ones(N))+np.outer(np.ones(M),(y-yf)**2);
    lensfunc = np.exp(-1j*2*np.pi/wlength*sq/2/focal_length)*D;
    lensfunc = tf.convert_to_tensor(lensfunc, dtype=tf.complex64)
    lensfunc = tf.reshape(lensfunc,shape=[1,M*N],name='lens')
    return lensfunc

def ideallens(wlength,focal_length,center,fshift,diameter,dx,dy,M,N):
    
    """
    Create function of a ideal paraxial lens
    INPUTS : 
        wlength -> wavelength of the optical signal
        f -> 2-by-1 list contains -> focal length in x and focal length in y
        center -> 2-by-1 list contains -> central shift of whole lens func. in x and y
        fshift -> lateral off-axis shift on the focal plane
        dia -> diameter of lens aperture
        dx,dy -> sampling intervals in space
        M,N -> size of simulation window
        
    OUTPUTS:
        lensfunc -> numerical lensfunction defined in space
        
    """
    x = (np.arange(M)-(M-1)/2)*dx
    y = (np.arange(N)-(N-1)/2)*dy
    x0,y0 = center[0], center[1]
    xf,yf = fshift[0], fshift[1]
    Lxy2 = (np.outer((x-x0)**2,np.ones(N))+np.outer(np.ones(M),(y-y0)**2))
    D = Lxy2<=(diameter**2)
    D = np.astype('int')
    sq = np.outer((x-xf)**2,np.ones(N))+np.outer(np.ones(M),(y-yf)**2);
    R= (sq+focal_length**2)**(0.5)
    lensfunc = np.exp(-1j*2*np.pi/wlength*R)*D;
    lensfunc = tf.convert_to_tensor(lensfunc, dtype=tf.complex64)
    lensfunc = tf.reshape(lensfunc,shape=[1,M*N],name='lens')
    return lensfunc

def batch_propagate(field, wlength, z, dx, dy, refidx,*theta0):

    if theta0:
        prop_field = tf_FSPAS_FFT(field, wlength, z, dx, dy, refidx,theta0)
    else:
        prop_field = tf_FSPAS_FFT(field, wlength, z, dx, dy, refidx)
#    img_prop, H = tf_FSPAS(img_reshaped[0, :, :], wlength, z, dx, dy, refidx)
#    img_prop_cat = tf.expand_dims(img_prop, 0)
#    for i in range(1, batch_shape[0]):
#        img_prop, H = tf_FSPAS(img_reshaped[i, :, :], wlength, z, dx, dy, refidx)
#        img_prop = tf.expand_dims(img_prop, 0)
#        img_prop_cat = tf.concat([img_prop_cat, img_prop], 0)
#
#    img_prop_reshaped = tf.reshape(img_prop_cat, [batch_shape[0], batch_shape[1]*batch_shape[2]])
#    img_prop_reshaped = tf.reshape(img_prop, [batch_shape[0], batch_shape[1]*batch_shape[2]])
    
    return prop_field