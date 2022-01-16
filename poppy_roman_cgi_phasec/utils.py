import poppy
from poppy.poppy_core import PlaneType
from poppy import accel_math

import proper
import astropy.io.fits as fits
import astropy.units as u
import numpy as np
import scipy
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from IPython.display import display, clear_output

import os
import copy
import time
import glob
from pathlib import Path

from .import polmap

from importlib import reload
import misc

def glass_index( glass, lambda_m, cgi_dir ):
    a = np.loadtxt( str( cgi_dir/'glass'/(glass+'_index.txt') ) )  # lambda_um index pairs
    f = interp1d( a[:,0], a[:,1], kind='cubic' )
    return f( lambda_m*1e6 )

def make_inwave(cgi_dir, D, lambda_c_m, lambda_m, npix, oversample, offsets, polaxis, display=True):
    ''' 
    Initialize the monochromatic input wavefront
        mode: 'HLC575', 'SPC730', or 'SPC825'
        cgi_dir: the directory to the POPPY CGI data files
        D: pupil diameter of the intended FresnelOpticalSystem and therefore the input wavefront
        lambda_c_m: central wavelength of the mode in units of m
        lambda_m: intended wavelength of the wavefront
        npix: number of pixels across the pupil diameter
        oversample: oversample factor of the wavefront
        offsets: source offsets given a tuple as such (x_offset, y_offset)
        polaxis: polarization axis to determine polarization aberrations with the polmap function
        display: display the input wavefront or not
    ''' 
    wfin = poppy.FresnelWavefront(beam_radius=D/2, wavelength=lambda_m, npix=npix, oversample=oversample)
    
    if polaxis!=0: 
        print('Employing polarization aberrations with polaxis={:d}.'.format(polaxis))
        polfile = cgi_dir/'pol'/'new_toma'
        polmap.polmap( wfin, polfile, npix, polaxis )
    else: print('Not employing polarization aberrations.')
    
    xoffset = offsets[0]
    yoffset = offsets[1]
    xoffset_lam = xoffset * (lambda_c_m / lambda_m).value # maybe use negative sign
    yoffset_lam = yoffset * (lambda_c_m / lambda_m).value 
    n = int(round(npix*oversample))
    x = np.tile( (np.arange(n)-n//2)/(npix/2.0), (n,1) )
    y = np.transpose(x)
    wfin.wavefront = wfin.wavefront * np.exp(complex(0,1) * np.pi * (xoffset_lam * x + yoffset_lam * y))
    
    if display:
        misc.myimshow2(np.abs(wfin.wavefront)**2, np.angle(wfin.wavefront),
                       'Input Wave Intensity', 'Input Wave Phase',
                       pxscl=wfin.pixelscale, 
                       cmap1='gist_heat', cmap2='viridis')
    
    return wfin


def resample_psf(pop_psf, pixelscale, npix): 
    ''' 
    Resample a POPPY PSF by providing the following:
        pop_psf: the FresnelWavefront object of the PSF
        pixelscale: the desired output pixelscale in units of m/pix
        npix: the desired output dimension of the PSF
    '''
    pop_psf_wf = pop_psf.wavefront
    pop_pxscl = pop_psf.pixelscale.value
    print('Input POPPY wavefront pixelscale: ', pop_pxscl*u.m/u.pix)
    mag = pop_pxscl/pixelscale
    resamp_pop_psf_wf = proper.prop_magnify(pop_psf_wf, mag, npix, AMP_CONSERVE=True)
    print('Interpolated POPPY wavefront pixelscale: ', pop_pxscl/mag*u.m/u.pix)
    return resamp_pop_psf_wf, mag


# convenient function for saving all wavefront data at each optic of the system once a PSF is calculated
def save_waves(wfs, use_opds, use_apertures, wfdir, npix=1000):
    if use_apertures==False and use_opds==False:
        optics = ['pupil', 'primary', 'secondary', 'pomafold', 'm3', 'm4', 'm5', 'ttfold', 'fsm', 'oap1', 
                  'focm', 'oap2', 'dm1', 'dm2', 'oap3', 'fold3', 'oap4', 'spm', 'oap5', 'fpm', 'oap6',
                  'lyotstop', 'oap7', 'fieldstop', 'oap8', 'filter', 'lens1', 'lens2', 'fold4', 'image']
        print('Saving wavefronts: ')
        for i,wf in enumerate(wfs):
            wavefront = misc.pad_or_crop(wf.wavefront, npix)

            wf_data = np.zeros(shape=(2,npix,npix))
            wf_data[0,:,:] = np.abs(wavefront)**2
            wf_data[1,:,:] = np.angle(wavefront)

            wf_fpath = wfdir/('wf_' + optics[i] + '_poppy' + '.fits')
            hdr = fits.Header()
            hdr['PIXELSCL'] = wf.pixelscale.value

            wf_hdu = fits.PrimaryHDU(wf_data, header=hdr)
            wf_hdu.writeto(wf_fpath, overwrite=True)
            print(i, 'Saved '+optics[i]+' wavefront to ' + str(wf_fpath))
    elif use_apertures==False and use_opds==True:
        optics = ['pupil', 
                  'primary', 'primary_opd', 
                  'secondary', 'secondary_opd',
                  'pomafold', 'pomafold_opd',
                  'm3', 'm3_opd',
                  'm4', 'm4_opd',
                  'm5', 'm5_opd',
                  'ttfold', 'ttfold_opd',
                  'fsm', 'fsm_opd',
                  'oap1', 'oap1_opd',
                  'focm', 'focm_opd',
                  'oap2', 'oap2_opd',
                  'dm1', 'dm1_opd',
                  'dm2', 'dm2_opd',
                  'oap3', 'oap3_opd',
                  'fold3', 'fold3_opd',
                  'oap4', 'oap4_opd',
                  'spm', 'spm_opd',
                  'oap5', 'oap5_opd',
                  'fpm', 
                  'oap6', 'oap6_opd',
                  'lyotstop', 
                  'oap7', 'oap7_opd',
                  'fieldstop', 
                  'oap8', 'oap8_opd',
                  'filter', 'filter_opd',
                  'lens1', 'lens_opd',
                  'lens2',
                  'fold4',
                  'image']
        print('Saving wavefronts: ')
        for i,wf in enumerate(wfs):
            wavefront = misc.pad_or_crop(wf.wavefront, npix)

            wf_data = np.zeros(shape=(2,npix,npix))
            wf_data[0,:,:] = np.abs(wavefront)**2
            wf_data[1,:,:] = np.angle(wavefront)

            wf_fpath = Path(wfdir)/('wf_' + optics[i] + '_poppy' + '.fits')
            hdr = fits.Header()
            hdr['PIXELSCL'] = wf.pixelscale.value

            wf_hdu = fits.PrimaryHDU(wf_data, header=hdr)
            wf_hdu.writeto(wf_fpath, overwrite=True)
            print(i, 'Saved '+optics[i]+' wavefront to ' + str(wf_fpath))
            
    print('All wavefronts saved.')
    
def display_dm(dm):
    fig, ax = plt.subplots(1,1,figsize=(5,5),dpi=125)
    dm.display(what='opd', opd_vmax=3e-7)
    plt.close()
    display(fig)
    
    
    
    