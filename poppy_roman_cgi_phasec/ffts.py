#   Copyright 2019 California Institute of Technology
# ------------------------------------------------------------------

# experience has shown that with the Intel math libraries the program
# can hang if run in a multithreaded condition with more than one 
# thread allocated, so set to 1. This must be done before importing numpy
import os
os.environ['MKL_NUM_THREADS'] = '1' 
import numpy as np

import proper

def ffts( wavefront, direction ):
    if wavefront.dtype != 'complex128' and wavefront.dtype != 'complex64':
        wavefront = wavefront.astype(complex)

    n = wavefront.shape[0]  # assumed to be power of 2
    wavefront[:,:] = np.roll( np.roll(wavefront, -n//2, 0), -n//2, 1 )  # shift to corner
    
    if proper.use_fftw:
        proper.prop_load_fftw_wisdom( n, proper.fftw_multi_nthreads ) 
        if direction == -1:
            proper.prop_fftw( wavefront, directionFFTW='FFTW_FORWARD' ) 
            wavefront /= np.size(wavefront)
        else:
            proper.prop_fftw( wavefront, directionFFTW='FFTW_BACKWARD' ) 
            wavefront *= np.size(wavefront)
    else:
        if direction == -1:
            wavefront[:,:] = np.fft.fft2(wavefront) / np.size(wavefront)
        else:
            wavefront[:,:] = np.fft.ifft2(wavefront) * np.size(wavefront)
    
    wavefront[:,:] = np.roll( np.roll(wavefront, n//2, 0), n//2, 1 )    # shift to center 

    return wavefront

