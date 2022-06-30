import numpy as np
import cupy as cp
from astropy.io import fits
import astropy.units as u
import time
from pathlib import Path
from scipy.interpolate import interp1d
import copy

import poppy
from poppy.poppy_core import PlaneType
from poppy import accel_math

import ray

cgi_dir = Path('/groups/douglase/kians-data-files/roman-cgi-phasec-data')
dm_dir = cgi_dir/'dm-acts'


# utility functions
def glass_index(wavelength, glass):
    a = np.loadtxt( str( cgi_dir/'glass'/(glass+'_index.txt') ) )  # lambda_um index pairs
    f = interp1d( a[:,0], a[:,1], kind='cubic' )
    return f( wavelength.value*1e6 )

@ray.remote
def run(cgi_mode='hlc', 
         wavelength=None, 
         npsf=64, psf_pixelscale=13e-6*u.m/u.pix, psf_pixelscale_lamD=None, 
        interp_order=3,
         offset=(0,0), 
         use_fpm=True, 
         use_fieldstop=True, 
         use_pupil_defocus=True, 
         use_opds=False, 
         dm1_ref=np.zeros((48,48)),
         dm2_ref=np.zeros((48,48)),
         polaxis=0,
        return_intermediates=False):   
    
    pupil_diam = 2.363114*u.m
    wavelength_c = 575e-9*u.m
    npix = 310
    oversample = 1024/310
    
    if wavelength is None: 
        wavelength = wavelength_c
    else: 
        wavelength = wavelength
    
    if psf_pixelscale_lamD is not None: # overrides psf_pixelscale this way
        psf_pixelscale_lamD = psf_pixelscale_lamD
        psf_pixelscale = 13e-6*u.m/u.pix / (0.5e-6/wavelength_c.value) * psf_pixelscale_lamD/0.5
    else:
        psf_pixelscale = psf_pixelscale
        psf_pixelscale_lamD = 1/2 * 0.5e-6/wavelength_c.value * psf_pixelscale.to(u.m/u.pix).value/13e-6
    detector = poppy.Detector(pixelscale=psf_pixelscale, fov_pixels=npsf, interp_order=interp_order)
    
    optics_dir = cgi_dir/'hlc'
    PUPIL = poppy.FITSOpticalElement('Roman Pupil', 
                                          transmission=str(optics_dir/'pupil_n310_new.fits'),
                                          pixelscale=pupil_diam.value / 310,
#                                                   rotation=180, 
#                                                   shift_x=-self.pupil_diam.value / 310,
#                                                   shift_y=-self.pupil_diam.value / 310,
                                          planetype=PlaneType.pupil)
    SPM = poppy.ScalarTransmission('SPM Plane (No Optic)', planetype=PlaneType.pupil)
    FPM_plane = poppy.ScalarTransmission('FPM Plane (No Optic)', planetype=PlaneType.intermediate) # placeholder
    if use_fpm:
        # Find nearest available FPM wavelength that matches specified wavelength and initialize the FPM data
        lam_um = wavelength.value * 1e6
        f = open( str(optics_dir/'fpm_files.txt') )
        fpm_nlam = int(f.readline())
        fpm_lams = np.zeros((fpm_nlam),dtype=float)
        for j in range(0,fpm_nlam): 
            fpm_lams[j] = float(f.readline())*1e-6
        fpm_root_fnames = [j.strip() for j in f.readlines()] 
        f.close()

        diff = np.abs(fpm_lams - wavelength.value)
        w = np.argmin( diff )
        if diff[w] > 0.1e-9: 
            raise Exception('Only wavelengths within 0.1nm of avalable FPM wavelengths can be used.'
                            'Closest available to requested wavelength is {}.'.format(fpm_lams[w]))
        fpm_rootname = optics_dir/fpm_root_fnames[w]

        fpm_r_fname = str(fpm_rootname)+'real.fits'
        fpm_i_fname = str(fpm_rootname)+'imag.fits'

        fpm_r = fits.getdata(fpm_r_fname)
        fpm_i = fits.getdata(fpm_i_fname)

        fpm_phasor = fpm_r + 1j*fpm_i

        print(fpm_r.shape)
#                 self.fpm_mask = (fpm_r != fpm_r[0,0]).astype(int)
        fpm_mask = (fpm_r != fpm_r[fpm_r.shape[0]-1,fpm_r.shape[0]-1]).astype(int)
        fpm_ref_wavelength = fits.getheader(fpm_r_fname)['WAVELENC']
        fpm_pixelscale_lamD = fits.getheader(fpm_r_fname)['PIXSCLLD']

    LS = poppy.FITSOpticalElement('Lyot Stop', 
                                       transmission=str(optics_dir/'lyot_hlc_n310_new.fits'), 
                                       pixelscale=5.50105901118828e-05 * 309/310,
#                                                rotation=180,
#                                                shift_x=-5.50105901118828e-05 * 309/310,
#                                                shift_y=-5.50105901118828e-05 * 309/310,
                                       planetype=PlaneType.pupil)

    if use_fieldstop: 
#                 radius = 9.7/(310/(self.npix*self.oversample)) * (self.wavelength_c/self.wavelength) * 7.229503001768824e-06*u.m
        radius = 9.7*oversample * (wavelength_c/wavelength) * 7.229503001768824e-06*u.m * 1.0
        fieldstop = poppy.CircularAperture(radius=radius, name='HLC Field Stop', gray_pixel=True)
    else: 
        fieldstop = poppy.ScalarTransmission(planetype=PlaneType.intermediate, name='Field Stop Plane (No Optic)')
    
    Nact = 48
    dm_diam = 46.3*u.mm
    act_spacing = 0.9906*u.mm

    DM1 = poppy.ContinuousDeformableMirror(dm_shape=(Nact,Nact), name='DM1', 
                                                actuator_spacing=act_spacing, 
#                                                     radius=dm_diam/2,
#                                                     inclination_x=0,inclination_y=9.65,
                                                inclination_x=9.65,inclination_y=0,
                                                influence_func=str(dm_dir/'proper_inf_func.fits'))
    DM2 = poppy.ContinuousDeformableMirror(dm_shape=(Nact,Nact), name='DM2', 
                                                actuator_spacing=act_spacing, 
#                                                 radius=dm_diam/2,
#                                                     inclination_x=0,inclination_y=9.65,
                                                inclination_x=9.65,inclination_y=0,
                                                influence_func=str(dm_dir/'proper_inf_func.fits'))
    DM1.set_surface(dm1_ref)
    DM2.set_surface(dm2_ref)
    
    # Define various optic focal lengths, diameters, and distances between optics.
    fl_pri = 2.838279206904720*u.m
    sm_despace_m = 0*u.m
    d_pri_sec = 2.285150508110035*u.m + sm_despace_m
    fl_sec = -0.654200796568004*u.m
    diam_sec = 0.58166*u.m
    d_sec_pomafold = 2.993753469304728*u.m + sm_despace_m
    diam_pomafold = 0.09*u.m
    d_pomafold_m3 = 1.680935841598811*u.m
    fl_m3 = 0.430216463069001*u.m
    diam_m3 = 0.2*u.m
    d_m3_pupil = 0.469156807765176*u.m
    d_m3_m4 = 0.943514749358944*u.m
    fl_m4 = 0.116239114833590*u.m
    diam_m4 = 0.07*u.m
    d_m4_m5 = 0.429145636743193*u.m
    fl_m5 = 0.198821518772608*u.m
    d_m5_pupil = 0.716529242927776*u.m
    diam_m5 = 0.07*u.m
    d_m5_ttfold = 0.351125431220770*u.m
    diam_ttfold = 0.06*u.m
    d_ttfold_fsm = d_m5_pupil - d_m5_ttfold 
    if use_pupil_defocus: d_ttfold_fsm = d_ttfold_fsm + 0.033609*u.m  # 33.6 mm to put pupil 6 mm from SPC mask
    diam_fsm = 0.0508*u.m
    d_fsm_oap1 = 0.354826767220001*u.m
    fl_oap1 = 0.503331895563883*u.m
    diam_oap1 = 0.060*u.m
    focm_z_shift_m = 0*u.m
    d_oap1_focm = 0.768029932093727*u.m + focm_z_shift_m
    diam_focm = 0.035*u.m
    d_focm_oap2 = 0.314507535543064*u.m + focm_z_shift_m
    fl_oap2 = 0.579205571254990*u.m
    diam_oap2 = 0.060*u.m
    d_oap2_dm1 = 0.775857408587825*u.m
    d_dm1_dm2 = 1.0*u.m
    d_dm2_oap3 = 0.394833855161549*u.m
    fl_oap3 = 1.217276467668519*u.m
    diam_oap3 = 0.06*u.m
    d_oap3_fold3 = 0.505329955078121*u.m
    diam_fold3 = 0.06*u.m
    d_fold3_oap4 = 1.158897671642761*u.m
    fl_oap4 = 0.446951159052363*u.m
    diam_oap4 = 0.06*u.m
    d_oap4_pupilmask = 0.423013568764728*u.m
    d_pupilmask_oap5 = 0.408810704327559*u.m
    fl_oap5 = 0.548189354706822*u.m
    diam_oap5 = 0.06*u.m
    d_oap5_fpm = fl_oap5                    # to front of FPM 
    fpm_thickness = 0.006363747896388863*u.m    # account for FPM thickness (inclination included)
    fpm_index = glass_index(wavelength, 'SILICA')
    d_fpm_oap6 = fpm_thickness / fpm_index + 0.543766629917668*u.m     # from front of FPM
    fl_oap6 = d_fpm_oap6
    diam_oap6 = 0.054*u.m
    d_oap6_lyotstop = 0.687476361491529*u.m
    d_oap6_exitpupil = d_oap6_lyotstop - 6e-3*u.m
    d_lyotstop_oap7 = 0.401748561745987*u.m
    fl_oap7 = 0.708251420923810*u.m
    diam_oap7 = 0.054*u.m
    d_oap7_fieldstop = fl_oap7 
    d_fieldstop_oap8 = 0.210985170345932*u.m * 0.997651
    fl_oap8 = d_fieldstop_oap8
    diam_oap8 = 0.026*u.m
    d_oap8_pupil = 0.237561587674008*u.m
    d_pupil_filter = 0.130*u.m
    d_oap8_filter = d_oap8_pupil + d_pupil_filter   # to front of filter
    diam_filter = 0.009*u.m
    filter_thickness = 0.004016105782012525*u.m      # account for filter thickness (inclination included)
    filter_index = glass_index(wavelength, 'SILICA')
    d_filter_lens = filter_thickness / filter_index + 0.210581269256657095*u.m  # from front of filter
    diam_lens = 0.0104*u.m
    d_lens_fold4 = 0.202432155667761*u.m
#     if use_pupil_lens != 0: d_lens_fold4 = d_lens_fold4 - 0.0002*u.m # from back of pupil imaging lens
#     elif use_defocus_lens != 0: d_lens_fold4 = d_lens_fold4 + 0.001*u.m # doublet is 1 mm longer than singlet, so make up for it
    diam_fold4 = 0.036*u.m
    d_fold4_image = 0.050000152941020161*u.m
    
    # define parameters related to final imaging lens, which is an air-gapped doublet
    lens_1_index = glass_index(wavelength, 'S-BSL7R')
    lens_2_index = glass_index(wavelength, 'PBM2R')
    r11, r12, lens_1_t = (0.10792660718579995*u.m, -0.10792660718579995*u.m, 0.003*u.m)
    r21, r22, lens_2_t = (1e10*u.m, 0.10608379812011390*u.m, 0.0025*u.m)
    air_gap = 0.0005*u.m
    
    fl_1 = 1 / ( (lens_1_index - 1) * ( 1.0/r11 - 1.0/r12 + (lens_1_index - 1)*lens_1_t / (lens_1_index*r11*r12) ) )
    d_pp_11 = -fl_1*(lens_1_index - 1)*lens_1_t / (lens_1_index*r12)
    d_pp_12 = -fl_1*(lens_1_index - 1)*lens_1_t / (lens_1_index*r11)

    fl_2 = 1 / ( (lens_2_index - 1) * ( 1.0/r21 - 1.0/r22 + (lens_2_index - 1)*lens_2_t / (lens_2_index*r21*r22) ) )
    d_pp_21 = -fl_2*(lens_2_index - 1)*lens_2_t / (lens_2_index*r22)
    d_pp_22 = -fl_2*(lens_2_index - 1)*lens_2_t / (lens_2_index*r21)

    d_filter_lens_1_pp1 = d_filter_lens + d_pp_11 
    d_lens_1_pp2_lens_2_pp1 = -d_pp_12 + air_gap + d_pp_21
    d_lens_2_pp2_fold4 = -d_pp_22 + d_lens_fold4

    # Define optical elements
    primary = poppy.QuadraticLens(fl_pri, name='Primary')
    secondary = poppy.QuadraticLens(fl_sec, name='Secondary')
    poma_fold = poppy.CircularAperture(radius=diam_pomafold/2, name="POMA_Fold")
    m3 = poppy.QuadraticLens(fl_m3, name='M3')
    m4 = poppy.QuadraticLens(fl_m4, name='M4')
    m5 = poppy.QuadraticLens(fl_m5, name='M5')
    tt_fold = poppy.CircularAperture(radius=diam_ttfold/2,name="TT_Fold")
    fsm = poppy.ScalarTransmission(planetype=PlaneType.intermediate, name='FSM')
    oap1 = poppy.QuadraticLens(fl_oap1, name='OAP1')
    focm = poppy.ScalarTransmission(planetype=PlaneType.intermediate, name='FOCM')
    oap2 = poppy.QuadraticLens(fl_oap2, name='OAP2')
    oap3 = poppy.QuadraticLens(fl_oap3, name='OAP3')
    fold3 = poppy.CircularAperture(radius=diam_fold3/2,name="Fold3")
    oap4 = poppy.QuadraticLens(fl_oap4, name='OAP4')
    oap5 = poppy.QuadraticLens(fl_oap5, name='OAP5')
    oap6 = poppy.QuadraticLens(fl_oap6, name='OAP6')
    oap7 = poppy.QuadraticLens(fl_oap7, name='OAP7')
    oap8 = poppy.QuadraticLens(fl_oap8, name='OAP8')
    filt = poppy.CircularAperture(radius=diam_filter/2, name='Filter')
    lens_1 = poppy.QuadraticLens(fl_1, name='LENS 1') # first lens of the doublet
    lens_2 = poppy.QuadraticLens(fl_2, name='LENS 2')
    fold4 = poppy.CircularAperture(radius=diam_fold4/2,name="Fold4")
        
    # Create the first part of the optical system
    fosys1 = poppy.FresnelOpticalSystem(name='HLC Pre-FPM', pupil_diameter=pupil_diam, 
                                        npix=npix, beam_ratio=1/oversample, verbose=True)
    
    fosys1.add_optic(PUPIL)
    fosys1.add_optic(primary)
    if use_opds: fosys1.add_optic(primary_opd)
        
    fosys1.add_optic(secondary, distance=d_pri_sec)
    if use_opds: fosys1.add_optic(secondary_opd)
        
    fosys1.add_optic(poma_fold, distance=d_sec_pomafold)
    if use_opds: fosys1.add_optic(poma_fold_opd)
        
    fosys1.add_optic(m3, distance=d_pomafold_m3)
    if use_opds: fosys1.add_optic(m3_opd)
        
    fosys1.add_optic(m4, distance=d_m3_m4)
    if use_opds: fosys1.add_optic(m4_opd)
        
    fosys1.add_optic(m5, distance=d_m4_m5)
    if use_opds: fosys1.add_optic(m5_opd)
        
    fosys1.add_optic(tt_fold, distance=d_m5_ttfold)
    if use_opds: fosys1.add_optic(tt_fold_opd)
        
    fosys1.add_optic(fsm, distance=d_ttfold_fsm)
    if use_opds: fosys1.add_optic(fsm_opd)
        
    fosys1.add_optic(oap1, distance=d_fsm_oap1)
    if use_opds: fosys1.add_optic(oap1_opd)
        
    fosys1.add_optic(focm, distance=d_oap1_focm)
    if use_opds: fosys1.add_optic(focm_opd)
        
    fosys1.add_optic(oap2, distance=d_focm_oap2)
    if use_opds: fosys1.add_optic(oap2_opd)
        
    fosys1.add_optic(DM1, distance=d_oap2_dm1)
    if use_opds: fosys1.add_optic(dm1_opd)
        
    fosys1.add_optic(DM2, distance=d_dm1_dm2)
    if use_opds: fosys1.add_optic(dm2_opd)
        
    fosys1.add_optic(oap3, distance=d_dm2_oap3)
    if use_opds: fosys1.add_optic(oap3_opd)
        
    fosys1.add_optic(fold3, distance=d_oap3_fold3)
    if use_opds: fosys1.add_optic(fold3_opd)
        
    fosys1.add_optic(oap4, distance=d_fold3_oap4)
    if use_opds: fosys1.add_optic(oap4_opd)
        
    fosys1.add_optic(SPM, distance=d_oap4_pupilmask)
    if use_opds: fosys1.add_optic(pupil_fold_opd)
        
    fosys1.add_optic(oap5, distance=d_pupilmask_oap5)
    if use_opds: fosys1.add_optic(oap5_opd)
        
    fosys1.add_optic(FPM_plane, distance=d_oap5_fpm)
    
    # Create second part of the optical system
    fosys2 = poppy.FresnelOpticalSystem(name='HLC Post-FPM', npix=npix, beam_ratio=1/oversample, verbose=True)
    
    fosys2.add_optic(FPM_plane)
    fosys2.add_optic(oap6, distance=d_fpm_oap6)
    if use_opds: fosys2.add_optic(oap6_opd)
        
    fosys2.add_optic(LS, distance=d_oap6_lyotstop)
    
    fosys2.add_optic(oap7, distance=d_lyotstop_oap7)
    if use_opds: fosys2.add_optic(oap7_opd)
        
    fosys2.add_optic(fieldstop, distance=d_oap7_fieldstop)
    
    fosys2.add_optic(oap8, distance=d_fieldstop_oap8)
    if use_opds: fosys2.add_optic(oap8_opd)
        
    fosys2.add_optic(filt, distance=d_oap8_filter)
    if use_opds: fosys2.add_optic(filter_opd)
        
    fosys2.add_optic(lens_1, distance=d_filter_lens_1_pp1)
    if use_opds: fosys2.add_optic(lens_opd)
        
    fosys2.add_optic(lens_2, distance=d_lens_1_pp2_lens_2_pp1)
    
    fosys2.add_optic(fold4, distance=d_lens_2_pp2_fold4)

    fosys2.add_optic(detector, distance=d_fold4_image)

#     fosys2.add_optic(HLC.detector, distance=d_lens_2_pp2_fold4 + d_fold4_image)
    
    # Calculate a psf from the first optical system to retrieve the final wavefront at the FPM plane 
    inwave = poppy.FresnelWavefront(beam_radius=pupil_diam/2, wavelength=wavelength,
                                    npix=npix, oversample=oversample)
    if polaxis!=0: 
        polfile = cgi_dir/'pol'/'phasec_pol'
        polmap.polmap( inwave, str(polfile), npix, polaxis )

    if offset[0]>0 or offset[1]>0:
        as_per_lamD = ((wavelength_c/pupil_diam)*u.radian).to(u.arcsec)
        inwave.tilt(Xangle=offset[0]*as_per_lamD, Yangle=offset[1]*as_per_lamD)
    
    fpm_hdu, wfs_to_fpm = fosys1.calc_psf(wavelength=wavelength, inwave=inwave, 
                                          return_final=True, return_intermediates=return_intermediates)
    inwave2 = copy.deepcopy(wfs_to_fpm[-1]) # copy Wavefront object for use in the post FPM system
    
    if use_fpm: 
        fpm_pxscl_lamD = ( fpm_pixelscale_lamD * wavelength_c.to(u.meter) / wavelength.to(u.meter) ).value
        nfpm = fpm_phasor.shape[0]
        n = inwave2.wavefront.shape[0]
        nfpmlamD = nfpm*fpm_pxscl_lamD*inwave2.oversample
        mft = poppy.matrixDFT.MatrixFourierTransform(centering='FFTSTYLE')

        # Apply the FPM with MFTs
        inwave2.wavefront = accel_math._ifftshift(inwave2.wavefront)
        inwave2.wavefront = accel_math.fft_2d(inwave2.wavefront, forward=False, fftshift=True) # do a forward FFT to virtual pupil
        inwave2.wavefront *= fpm_phasor[-1,-1]
        wavefront_fpm = mft.perform(inwave2.wavefront, nfpmlamD, nfpm) # MFT back to highly sampled focal plane
        
        if poppy.accel_math._USE_CUPY:
            wavefront_fpm *= cp.array(fpm_mask * (fpm_phasor - 1))
        else:
            wavefront_fpm *= HLC.fpm_mask * (HLC.fpm_phasor - 1)
            
        wavefront_fpm = mft.inverse(wavefront_fpm, nfpmlamD, n) # MFT to virtual pupil
        inwave2.wavefront += wavefront_fpm
        inwave2.wavefront = accel_math.fft_2d(inwave2.wavefront, forward=True, fftshift=True) # FFT back to normally sampled focus
        inwave2.wavefront = accel_math._fftshift(inwave2.wavefront)
        
    psf_hdu, wfs_from_fpm = fosys2.calc_psf(wavelength=wavelength, inwave=inwave2, normalize='none',
                                            return_final=True, return_intermediates=return_intermediates,)
    
    if return_intermediates:
        wfs_to_fpm.pop(-1)
        wfs = wfs_to_fpm + wfs_from_fpm
    else: 
        wfs = wfs_from_fpm[-1]

    return wfs



