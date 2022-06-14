import numpy as np
import cupy as cp
from astropy.io import fits
import astropy.units as u
import time
from pathlib import Path
from scipy.interpolate import interp1d

import warnings

import poppy
from poppy.poppy_core import PlaneType

import ray

from . import hlc, spc, polmap, hlc_v2, spc_v2

from importlib import reload
reload(hlc)
reload(spc)
reload(hlc_v2)
reload(spc_v2)
reload(polmap)

cgi_dir = Path('/groups/douglase/kians-data-files/roman-cgi-phasec-data')
dm_dir = Path('/groups/douglase/kians-data-files/roman-cgi-phasec-data/dm-acts')

class CGI():

    def __init__(self, 
                 cgi_mode='hlc', 
                 wavelength=None, 
                 npsf=64, psf_pixelscale=13e-6*u.m/u.pix, psf_pixelscale_lamD=None, interp_order=3,
                 offset=(0,0), 
                 use_fpm=True, use_fieldstop=True, 
                 use_pupil_defocus=True, use_opds=False, 
                 polaxis=0, 
                 return_intermediates=False):
        
        self.cgi_mode = cgi_mode
        
        self.pupil_diam = 2.363114*u.m
        if self.cgi_mode=='hlc': 
            self.wavelength_c = 575e-9*u.m
            self.npix = 310
            self.oversample = 1024/310
            self.D = self.pupil_diam*self.npix/309
        elif self.cgi_mode=='spc-spec':
            self.wavelength_c = 730e-9*u.m
            self.npix = 1000
            self.oversample = 2.048
            self.D = self.pupil_diam
        elif self.cgi_mode=='spc-wide':
            self.wavelength_c = 825e-9*u.m
            self.npix = 1000
            self.oversample = 2.048
#             self.oversample = 4.096
            self.D = self.pupil_diam
            
        self.as_per_lamD = ((self.wavelength_c/self.pupil_diam)*u.radian).to(u.arcsec)
        
        if wavelength is None: 
            self.wavelength = self.wavelength_c
        else: 
            self.wavelength = wavelength
        
        self.offset = offset
        self.use_fpm = use_fpm
        self.use_pupil_defocus = use_pupil_defocus
        self.use_fieldstop = use_fieldstop
        self.use_opds = use_opds
        self.polaxis = polaxis
        
        self.npsf = npsf
        if psf_pixelscale_lamD is not None: # overrides psf_pixelscale this way
            self.psf_pixelscale_lamD = psf_pixelscale_lamD
            self.psf_pixelscale = 13e-6*u.m/u.pix / (0.5e-6/self.wavelength_c.value) * self.psf_pixelscale_lamD/0.5
        else:
            self.psf_pixelscale = psf_pixelscale
            self.psf_pixelscale_lamD = 1/2 * 0.5e-6/self.wavelength_c.value * self.psf_pixelscale.to(u.m/u.pix).value/13e-6
        self.interp_order = interp_order # interpolation order for resampling wavefront at detector
        
        self.texp = 1*u.s
        self.peak_photon_flux = 1e8*u.photon/u.s
        
        self.detector_gain = 1*u.electron/u.photon
        self.read_noise_std = 1.7*u.electron/u.photon
        self.well_depth = 3e4*u.electron
        self.dark_rate = 0.015*u.electron/u.pix/u.s  # [e-/pixel/second]
        
        self.init_mode_optics()
        self.init_dms()
        if self.use_opds: 
            self.init_opds()
        
        self.return_intermediates = return_intermediates
        
    def copy_mode_settings(self, nactors=1):
        settings = []
        for i in range(nactors):
            settings.append({'cgi_mode':self.cgi_mode,
                             'wavelength':self.wavelength, 
                             'npsf':self.npsf, 
                             'psf_pixelscale':self.psf_pixelscale, 
                             'interp_order':self.interp_order,
                             'offset':self.offset,  
                             'use_fpm':self.use_fpm,
                             'use_fieldstop':self.use_fieldstop,
                             'use_pupil_defocus':self.use_pupil_defocus,
                             'use_opds':self.use_opds, 
                             'polaxis':self.polaxis,
                             'return_intermediates':self.return_intermediates})
        return settings
           
    def init_mode_optics(self):
        self.FPM_plane = poppy.ScalarTransmission('FPM Plane (No Optic)', planetype=PlaneType.intermediate) # placeholder
        
        if self.cgi_mode=='hlc':
            self.optics_dir = cgi_dir/'hlc'
            self.PUPIL = poppy.FITSOpticalElement('Roman Pupil', 
                                                  transmission=str(self.optics_dir/'pupil.fits'),
                                                  planetype=PlaneType.pupil)
            self.SPM = poppy.ScalarTransmission('SPM Plane (No Optic)', planetype=PlaneType.pupil)
            if self.use_fpm:
                # Find nearest available FPM wavelength that matches specified wavelength and initialize the FPM data
                lam_um = self.wavelength.value * 1e6
                f = open( str(self.optics_dir/'fpm_files.txt') )
                fpm_nlam = int(f.readline())
                fpm_lams = np.zeros((fpm_nlam),dtype=float)
                for j in range(0,fpm_nlam): 
                    fpm_lams[j] = float(f.readline())*1e-6
                fpm_root_fnames = [j.strip() for j in f.readlines()] 
                f.close()

                diff = np.abs(fpm_lams - self.wavelength.value)
                w = np.argmin( diff )
                if diff[w] > 0.1e-9: 
                    raise Exception('Only wavelengths within 0.1nm of avalable FPM wavelengths can be used.'
                                    'Closest available to requested wavelength is {}.'.format(fpm_lams[w]))
                fpm_rootname = self.optics_dir/fpm_root_fnames[w]

                fpm_r_fname = str(fpm_rootname)+'real.fits'
                fpm_i_fname = str(fpm_rootname)+'imag.fits'

                fpm_r = fits.getdata(fpm_r_fname)
                fpm_i = fits.getdata(fpm_i_fname)
                    
                self.fpm_phasor = fpm_r + 1j*fpm_i
                self.fpm_mask = (fpm_r != fpm_r[0,0]).astype(int)
                self.fpm_ref_wavelength = fits.getheader(fpm_r_fname)['WAVELENC']
                self.fpm_pixelscale_lamD = fits.getheader(fpm_r_fname)['PIXSCLLD']
            else:
                self.FPM = None
            
            self.LS = poppy.FITSOpticalElement('Lyot Stop', 
                                               transmission=str(self.optics_dir/'lyot_rotated.fits'), 
                                               planetype=PlaneType.pupil)
            
            if self.use_fieldstop: 
                radius = 9.7/(309/(self.npix*self.oversample)) * (self.wavelength_c/self.wavelength) * 7.229503001768824e-06*u.m
                self.fieldstop = poppy.CircularAperture(radius=radius, name='HLC Field Stop')
            else: 
                self.fieldstop = poppy.ScalarTransmission(planetype=PlaneType.intermediate, name='Field Stop Plane (No Optic)')
                
        elif self.cgi_mode=='spc-spec': 
            self.optics_dir = cgi_dir/'spc-spec'            
            self.PUPIL = poppy.FITSOpticalElement('Roman Pupil', 
#                                                   rotation=180*u.degree,
                                                  transmission=str(self.optics_dir/'pupil_SPC-20200617_1000.fits'),
                                                  planetype=PlaneType.pupil)
            self.SPM = poppy.FITSOpticalElement('SPM', 
#                                                 rotation=180*u.degree,
                                                transmission=str(self.optics_dir/'SPM_SPC-20200617_1000_rounded9_rotated.fits'),
                                                planetype=PlaneType.pupil)
            self.LS = poppy.FITSOpticalElement('Lyot Stop',
                                               transmission=str(self.optics_dir/'LS_SPC-20200617_1000.fits'), 
                                               planetype=PlaneType.pupil)
            if self.use_fpm: 
                self.FPM = poppy.FixedSamplingImagePlaneElement('FPM', 
                                                                transmission=str(self.optics_dir/'fpm_0.05lamD.fits'))
            else: 
                self.FPM = poppy.ScalarTransmission(name='FPM Plane (No Optic)', planetype=PlaneType.intermediate) 
            self.fieldstop = poppy.ScalarTransmission(planetype=PlaneType.intermediate, name='Field Stop Plane (No Optic)')
            self.use_fieldstop = False
            
        elif self.cgi_mode=='spc-wide':
            self.optics_dir = cgi_dir/'spc-wide'            
            self.PUPIL = poppy.FITSOpticalElement('Roman Pupil', 
                                                  transmission=str(self.optics_dir/'pupil_SPC-20200610_1000.fits'),
                                                  planetype=PlaneType.pupil)
            self.SPM = poppy.FITSOpticalElement('SPM', 
                                                str(self.optics_dir/'SPM_SPC-20200610_1000_rounded9_gray_rotated.fits'),
                                                planetype=PlaneType.pupil)
            if self.use_fpm: 
                self.FPM = poppy.FixedSamplingImagePlaneElement('FPM', 
                                                                str(self.optics_dir/'FPM_SPC-20200610_0.1_lamc_div_D.fits'))
            else: 
                self.FPM = poppy.ScalarTransmission(name='FPM Plane (No Optic)', planetype=PlaneType.intermediate) 
            
            self.LS = poppy.FITSOpticalElement('Lyot Stop',
                                               transmission=str(self.optics_dir/'LS_SPC-20200610_1000.fits'), 
                                               planetype=PlaneType.pupil)
            self.fieldstop = poppy.ScalarTransmission(planetype=PlaneType.intermediate, name='Field Stop Plane (No Optic)')
            self.use_fieldstop = False
            
        self.detector = poppy.Detector(pixelscale=self.psf_pixelscale, fov_pixels=self.npsf, interp_order=self.interp_order)
        
    # DM methods
    def init_dms(self):
        self.Nact = 48
        self.dm_diam = 46.3*u.mm
        self.act_spacing = 0.9906*u.mm
        
        
        
        self.DM1 = poppy.ContinuousDeformableMirror(dm_shape=(self.Nact,self.Nact), name='DM1', 
                                                    actuator_spacing=self.act_spacing, radius=self.dm_diam/2,
                                                    inclination_x=0,inclination_y=9.65,
                                                    influence_func=str(dm_dir/'proper_inf_func.fits'))
        self.DM2 = poppy.ContinuousDeformableMirror(dm_shape=(self.Nact,self.Nact), name='DM2', 
                                                    actuator_spacing=self.act_spacing, radius=self.dm_diam/2,
                                                    inclination_x=0,inclination_y=9.65,
                                                    influence_func=str(dm_dir/'proper_inf_func.fits'))
    
    def reset_dms(self):
        self.DM1.set_surface( np.zeros((self.Nact, self.Nact)) )
        self.DM2.set_surface( np.zeros((self.Nact, self.Nact)) )
            
    def set_dm1(self, dm_command):
        dm_command = self.check_dm_command_shape(dm_command)
        self.DM1.set_surface(dm_command)
    
    def set_dm2(self, dm_command):
        dm_command = self.check_dm_command_shape(dm_command)
        self.DM2.set_surface(dm_command)
        
    def add_dm1(self, dm_command):
        dm_command = self.check_dm_command_shape(dm_command)
        self.DM1.set_surface(self.DM1.surface.get() + dm_command) # I should make the DM.surface attribute be Numpy no matter what
        
    def add_dm2(self, dm_command):
        dm_command = self.check_dm_command_shape(dm_command)
        self.DM2.set_surface(self.DM2.surface.get() + dm_command)
    
    def check_dm_command_shape(self, dm_command):
        if dm_command.shape[0]==self.Nact**2 or dm_command.shape[1]==self.Nact**2: # passes if shape does not have 2 values
            dm_command = dm_command.reshape((self.Nact, self.Nact))
        return dm_command
    
    # utility functions
    def glass_index(self, glass):
        a = np.loadtxt( str( cgi_dir/'glass'/(glass+'_index.txt') ) )  # lambda_um index pairs
        f = interp1d( a[:,0], a[:,1], kind='cubic' )
        return f( self.wavelength.value*1e6 )

    def init_inwave(self):
        inwave = poppy.FresnelWavefront(beam_radius=self.D/2, wavelength=self.wavelength,
                                        npix=self.npix, oversample=self.oversample)
        if self.polaxis!=0: 
            polfile = cgi_dir/'pol'/'phasec_pol'
            polmap.polmap( inwave, str(polfile), self.npix, self.polaxis )
            
        if self.offset[0]>0 or self.offset[1]>0:
            inwave.tilt(Xangle=self.offset[0]*self.as_per_lamD, Yangle=self.offset[1]*self.as_per_lamD)
        
        inwave.w_0 = self.pupil_diam/2
        
        self.inwave = inwave
    
    
    def init_opds(self):
        opddir = cgi_dir/'opd-maps'

        opdunits = 'meters'

        self.primary_opd = poppy.FITSOpticalElement('Primary OPD', 
                                                    opd=str(opddir/'roman_phasec_PRIMARY_synthetic_phase_error_V1.0.fits'),
                                               opdunits=opdunits, planetype=PlaneType.intermediate)

        self.secondary_opd = poppy.FITSOpticalElement('Secondary OPD',
                                                 opd=str(opddir/'roman_phasec_SECONDARY_synthetic_phase_error_V1.0.fits'),
                                                 opdunits=opdunits, planetype=PlaneType.intermediate)

        self.poma_fold_opd = poppy.FITSOpticalElement('POMA-Fold OPD', 
                                                 opd=str(opddir/'roman_phasec_POMAFOLD_measured_phase_error_V1.1.fits'),
                                                 opdunits=opdunits,planetype=PlaneType.intermediate)

        self.m3_opd = poppy.FITSOpticalElement('M3 OPD', 
                                          opd=str(opddir/'roman_phasec_M3_measured_phase_error_V1.1.fits'), 
                                          opdunits=opdunits, planetype=PlaneType.intermediate)

        self.m4_opd = poppy.FITSOpticalElement('M4 OPD',
                                          opd=str(opddir/'roman_phasec_M4_measured_phase_error_V1.1.fits'), 
                                          opdunits=opdunits, planetype=PlaneType.intermediate)

        self.m5_opd = poppy.FITSOpticalElement('M5 OPD',
                                          opd=str(opddir/'roman_phasec_M5_measured_phase_error_V1.1.fits'),
                                          opdunits=opdunits, planetype=PlaneType.intermediate)

        self.tt_fold_opd = poppy.FITSOpticalElement('TT-Fold OPD', 
                                               opd=str(opddir/'roman_phasec_TTFOLD_measured_phase_error_V1.1.fits'), 
                                               opdunits=opdunits, planetype=PlaneType.intermediate)

        self.fsm_opd = poppy.FITSOpticalElement('FSM OPD', 
                                           opd=str(opddir/'roman_phasec_LOWORDER_phase_error_V2.0.fits'),
                                           opdunits=opdunits, planetype=PlaneType.intermediate)

        self.oap1_opd = poppy.FITSOpticalElement('OAP1 OPD',
                                            opd=str(opddir/'roman_phasec_OAP1_phase_error_V3.0.fits'),
                                            opdunits=opdunits, planetype=PlaneType.intermediate)

        self.focm_opd = poppy.FITSOpticalElement('FOCM OPD', 
                                            opd=str(opddir/'roman_phasec_FCM_EDU_measured_coated_phase_error_V2.0.fits'), 
                                            opdunits=opdunits, planetype=PlaneType.intermediate)

        self.oap2_opd = poppy.FITSOpticalElement('OAP2 OPD', 
                                            opd=str(opddir/'roman_phasec_OAP2_phase_error_V3.0.fits'), 
                                            opdunits=opdunits, planetype=PlaneType.intermediate)

        self.dm1_opd = poppy.FITSOpticalElement('DM1 OPD',
                                           opd=str(opddir/'roman_phasec_DM1_phase_error_V1.0.fits'), 
                                           opdunits=opdunits, planetype=PlaneType.intermediate)

        self.dm2_opd = poppy.FITSOpticalElement('DM2 OPD', 
                                           opd=str(opddir/'roman_phasec_DM2_phase_error_V1.0.fits'),
                                           opdunits=opdunits, planetype=PlaneType.intermediate)

        self.oap3_opd = poppy.FITSOpticalElement('OAP3 OPD',
                                            opd=str(opddir/'roman_phasec_OAP3_phase_error_V3.0.fits'), 
                                            opdunits=opdunits, planetype=PlaneType.intermediate)

        self.fold3_opd = poppy.FITSOpticalElement('F3 OPD',
                                             opd=str(opddir/'roman_phasec_FOLD3_FLIGHT_measured_coated_phase_error_V2.0.fits'), 
                                             opdunits=opdunits, planetype=PlaneType.intermediate)

        self.oap4_opd = poppy.FITSOpticalElement('OAP4 OPD', 
                                            opd=str(opddir/'roman_phasec_OAP4_phase_error_V3.0.fits'), 
                                            opdunits=opdunits, planetype=PlaneType.intermediate)

        self.pupil_fold_opd = poppy.FITSOpticalElement('SPM Fold OPD', 
                                                  opd=str(opddir/'roman_phasec_PUPILFOLD_phase_error_V1.0.fits'), 
                                                  opdunits=opdunits, planetype=PlaneType.intermediate)

        self.pupil_mask_opd = poppy.FITSOpticalElement('SPM OPD', 
                                                  opd=str(opddir/'roman_phasec_PUPILMASK_phase_error_V1.0.fits'), 
                                                  opdunits=opdunits, planetype=PlaneType.intermediate)

        self.oap5_opd = poppy.FITSOpticalElement('OAP5 OPD',
                                            opd=str(opddir/'roman_phasec_OAP5_phase_error_V3.0.fits'), 
                                            opdunits=opdunits, planetype=PlaneType.intermediate)

        self.oap6_opd = poppy.FITSOpticalElement('OAP6 OPD', 
                                            opd=str(opddir/'roman_phasec_OAP6_phase_error_V3.0.fits'), 
                                            opdunits=opdunits, planetype=PlaneType.intermediate)

        self.oap7_opd = poppy.FITSOpticalElement('OAP7 OPD', 
                                            opd=str(opddir/'roman_phasec_OAP7_phase_error_V3.0.fits'),
                                            opdunits=opdunits, planetype=PlaneType.intermediate)

        self.oap8_opd = poppy.FITSOpticalElement('OAP8 OPD', 
                                            opd=str(opddir/'roman_phasec_OAP8_phase_error_V3.0.fits'), 
                                            opdunits=opdunits, planetype=PlaneType.intermediate)

        self.filter_opd = poppy.FITSOpticalElement('Filter OPD',
                                              opd=str(opddir/'roman_phasec_FILTER_phase_error_V1.0.fits'), 
                                              opdunits=opdunits, planetype=PlaneType.intermediate)

        self.lens_opd = poppy.FITSOpticalElement('LENS OPD', 
                                            opd=str(opddir/'roman_phasec_LENS_phase_error_V1.0.fits'), 
                                            opdunits=opdunits, planetype=PlaneType.intermediate)
    
    def add_noise(self, image):
        peak_photons = self.peak_photon_flux * self.texp
        peak_electrons = self.detector_gain * peak_photons

        image_in_electrons = peak_electrons.value * image

        # Add photon shot noise
        if type(image) is np.ndarray:
            noisy_image_in_electrons = np.random.poisson(image_in_electrons)
        else:
            noisy_image_in_electrons = cp.random.poisson(image_in_electrons)

        # Compute dark current
        if type(image) is np.ndarray:
            dark_current = (self.dark_rate * self.texp).value * np.ones_like(image)
            dark_current = np.random.poisson(dark_current)
        else:
            dark_current = (self.dark_rate * self.texp).value * cp.ones_like(image)
            dark_current = cp.random.poisson(dark_current)

        # Compute Gaussian read noise
        if type(image) is np.ndarray:
            read_noise = self.read_noise_std.value * np.random.randn(image.shape[0], image.shape[1])
        else:
            read_noise = self.read_noise_std.value * cp.random.randn(image.shape[0], image.shape[1])

        # Convert back from e- to counts and then discretize
        if type(image) is np.ndarray:
            image_in_photons = np.round( (noisy_image_in_electrons + dark_current + read_noise) / self.detector_gain.value)
        else:
            image_in_photons = cp.round( (noisy_image_in_electrons + dark_current + read_noise) / self.detector_gain.value)

        # Convert back from counts to normalized intensity
        noisy_image = image_in_photons / self.peak_photon_flux.value

        return noisy_image

    def calc_psf(self, quiet=False):
        start = time.time()
        if not quiet: print('Propagating wavelength {:.3f}.'.format(self.wavelength.to(u.nm)))
            
        self.init_inwave()
        if self.cgi_mode=='hlc':
            wfs = hlc.run(self)
        else:
            wfs = spc.run(self)
            
        if not self.return_intermediates:
            wfs = wfs[-1]
            
        if not quiet: print('PSF calculated in {:.3f}s'.format(time.time()-start))
            
        return wfs

    def calc_psf2(self, quiet=False):
        start = time.time()
        if not quiet: print('Propagating wavelength {:.3f}.'.format(self.wavelength.to(u.nm)))
            
        self.init_inwave()
        if self.cgi_mode=='hlc':
            wfs = hlc_v2.run(self)
        else:
            wfs = spc_v2.run(self)
            
        if not self.return_intermediates:
            wfs = wfs[-1]
            
        if not quiet: print('PSF calculated in {:.3f}s'.format(time.time()-start))
            
        return wfs
    
CGIR = ray.remote(CGI)
        
def create_actors(ncpus=32, ngpus=1, settings=[{'cgi_mode':'hlc', 
                                                'wavelength':None, 
                                                'npsf':64, 
                                                'psf_pixelscale':13e-6*u.m/u.pix, 
                                                'psf_pixelscale_lamD':None, 
                                                'interp_order':3,
                                                'offset':(0,0), 
                                                'use_fpm':True,
                                                'use_fieldstop':True, 
                                                'use_pupil_defocus':True,
                                                'use_opds':False, 
                                                'polaxis':0, 
                                                'return_intermediates':False}]):
    
    actors = []
    for i in range(len(settings)):
        if isinstance(ncpus, list) or isinstance(ncpus, np.ndarray):
            num_cpus = ncpus[i]
        else: 
            num_cpus = ncpus
        if isinstance(ngpus, list) or isinstance(ngpus, np.ndarray):
            num_gpus = ngpus[i]
        else: 
            num_gpus = ngpus
            
        actors.append( CGIR.options(num_cpus=num_cpus, num_gpus=num_gpus).remote(**settings[i]) ) 
    
    return actors
    

def calc_psfs(actors, quiet=True):
    start = time.time()
    
    pending_wfs = []
    for i in range(len(actors)):
        future_wfs = actors[i].calc_psf.remote(quiet=quiet)
        pending_wfs.append(future_wfs)
    wfs = ray.get(pending_wfs)
    
    if not quiet: print('All PSFs calculated in {:.3f}s'.format(time.time()-start))
    return wfs



@ray.remote
def _calc_psf(CGI, quiet=False):
    start = time.time()
    if not quiet: print('Propagating wavelength {:.3f}.'.format(CGI.wavelength.to(u.nm)))

    CGI.init_inwave()
    if CGI.cgi_mode=='hlc':
        wfs = hlc.run(CGI)
    else:
        wfs = spc.run(CGI)

    if not CGI.return_intermediates:
        wfs = wfs[-1]

    if not quiet: print('PSF calculated in {:.3f}s'.format(time.time()-start))

    return wfs   




        