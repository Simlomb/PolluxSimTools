#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 12:31:11 2017
@author: gkanarek
@author pollux side: Simlomb (Simona Lombardo)
"""

from __future__ import (print_function, division, absolute_import, with_statement,
                        nested_scopes, generators)
import os
import numpy as np
import astropy.units as u
import astropy.constants as const
from scipy.ndimage import gaussian_filter
import pickle
import math

from syotools.models.base import PersistentModel
from syotools.defaults import default_exposure
from syotools.utils import pre_encode, pre_decode
from syotools.spectra import SpectralLibrary
from syotools.spectra.utils import renorm_sed

def nice_print(arr):
    """
    Utility to make the verbose output more readable.
    """
    
    arr = pre_decode(arr) #in case it's a JsonUnit serialization

    if isinstance(arr, u.Quantity):
        l = ['{:.2f}'.format(i) for i in arr.value]
    else:
        l = ['{:.2f}'.format(i) for i in arr]
    return ', '.join(l)

class Exposure(PersistentModel):
    """
    The base exposure class, which provides parameter storage for 
    optimization, and all exposure-specific calculations.
    
    This class encompasses both imaging and spectral exposures -- we will
    assume that all calculations are performed with access to the full SED 
    (which is simply interpolated to the correct wavebands for imaging).
    
    The SNR, exptime, and limiting magnitude can each be calculated from the
    other two. To trigger such calculations when parameters are updated, we
    will need to create property setters.
    
    Attributes:
        telescope    - the Telescope model instance associated with this exposure
        camera       - the Camera model instance associated with this exposure
        spectrograph - the Spectrograph model instance (if applicable) associated
                       with this exposure
    
        exp_id       - a unique exposure ID, used for save/load purposes (string)
                        NOTE: THIS HAS NO DEFAULT, A NEW EXP_ID IS CREATED
                        WHENEVER A NEW CALCULATION IS SAVED.
        sed_flux     - the spectral energy distribution of the target (float array)
        sed_wav      - the wavelengths associated with the SED flux (float array)
        sed_id       - for default (pysynphot) spectra, the id (i.e. the key 
                       into the default spectra dictionary), otherwise "user" (string)
        n_exp        - the desired number of exposures (integer)
        exptime      - the desired exposure time (float array)
        snr          - the desired S/N ratio (float array)
        magnitude    - either the input source magnitude, in which case this is
                       equal to the SED interpolated to the desired wavelengths,
                       or the limiting magnitude of the exposure (float array)
        redshift     - the redshift of the SED (float)
        unknown      - a flag to indicate which variable should be calculated
                       ('snr', 'exptime', or 'magnitude'). this should generally 
                       be set by the tool, and not be available to users. (string)
        
        _default_model - used by PersistentModel
    """
    
    _default_model = default_exposure
    
    telescope = None
    camera = None
    spectrograph = None
    spectropolarimeter = None
    exp_id = ''
    _sed = pre_encode(np.zeros(1, dtype=float) * u.ABmag) #default is set via sed_id
    _sed_id = ''
    n_exp = 0
    _exptime = pre_encode(np.zeros(1, dtype=float) * u.s)
    _snr = pre_encode(np.zeros(1, dtype=float) * u.dimensionless_unscaled)
    _final_image = pre_encode(np.zeros(1, dtype=float) * u.dimensionless_unscaled)
    _wave_image = pre_encode(np.zeros(1, dtype=float) * u.dimensionless_unscaled)
    _magnitude = pre_encode(np.zeros(1, dtype=float) * u.ABmag)
    _redshift = 0.
    _unknown = '' #one of 'snr', 'magnitude', 'exptime'
    
    verbose = False #set this for debugging purposes only
    _disable = False #set this to disable recalculating (when updating several attributes at the same time)
    
    def disable(self):
        self._disable = True
    
    def enable(self,make_image=False,photon_count=0,gain=1000.):
        self._disable = False
        self.calculate(make_image,photon_count,gain)
    
    #Property wrappers for the three possible uknowns, so that we can auto-
    #calculate whenever they're set, and to prevent overwriting previous
    #calculations by accident.
    
    @property
    def unknown(self):
        return self._unknown
    
    @unknown.setter
    def unknown(self, new_unknown):
        self._unknown = new_unknown
        self.calculate(make_image=False)
    
    def _ensure_array(self, quant):
        """
        Ensure that the given Quantity is an array, propagating if necessary.
        """
        q = pre_encode(quant)
        if len(q) < 2:
            import pdb; pdb.set_trace()
        val = q[1]['value']
        if not isinstance(val, list):
            if self.camera is None:
                nb = 1
            else:
                nb = self.recover('camera.n_bands')
            q[1]['value'] = np.full(nb, val).tolist()
        
        return q
    
    @property
    def exptime(self):
        return self._exptime
    
    @exptime.setter
    def exptime(self, new_exptime):
        if self.unknown == "exptime":
            return
        self._exptime = self._ensure_array(new_exptime)
        self.calculate(make_image=False)
    
    @property
    def snr(self):
        return self._snr

    @property
    def final_image(self):
        return self._final_image

    @property
    def wave_image(self):
        return self._wave_image
    
    @snr.setter
    def snr(self, new_snr):
        if self.unknown == "snr":
            return
        self._snr = self._ensure_array(new_snr)
        self.calculate(make_image=False)
    @final_image.setter
    def final_image(self, new_final_image):
        if self.unknown == "final_image":
            return
        self._final_image = self._ensure_array(new_final_image)
        self.calculate(make_image=False)

    @wave_image.setter
    def wave_image(self, new_wave_image):
        if self.unknown == "wave_image":
            return
        self._wave_image = self._ensure_array(new_wave_image)
        self.calculate(make_image=False)

    @property
    def sed(self):
        """
        Return a spectrum, redshifted if necessary. We don't just store the 
        redshifted spectrum because pysynphot doesn't save the original, it 
        just returns a new copy of the spectrum with redshifted wavelengths.
        """
        sed = pre_decode(self._sed)
        z = self.recover('redshift')
        return pre_encode(sed.redshift(z))
    
    @sed.setter
    def sed(self, new_sed):
        self._sed = pre_encode(new_sed)
        self.calculate(make_image=False)
        
    @property
    def sed_id(self):
        return self._sed_id
    
    @sed_id.setter
    def sed_id(self, new_sed_id):
        if new_sed_id == self._sed_id:
            return
        self._sed_id = new_sed_id
        self._sed = pre_encode(SpectralLibrary.get(new_sed_id, SpectralLibrary.fab))
        self.calculate(make_image=False)
        
    def renorm_sed(self, new_mag, bandpass='johnson,v'):
        sed = self.recover('_sed')
        self._sed = renorm_sed(sed, pre_decode(new_mag), bandpass=bandpass)
        self.calculate(make_image=False)
    
    @property
    def interpolated_sed(self):
        """
        The exposure's SED interpolated at the camera bandpasses.
        """
        if not self.camera:
            return self.sed
        sed = self.recover('sed')
        return pre_encode(self.camera.interpolate_at_bands(sed))
    
    @property
    def magnitude(self):
        if self.unknown == "magnitude":
            return self._magnitude
        #If magnitude is not unknown, it should be interpolated from the SED
        #at the camera bandpasses. 
        return self.interpolated_sed
    
    @magnitude.setter
    def magnitude(self, new_magnitude):
        if self.unknown == "magnitude":
            return
        self._magnitude = self._ensure_array(new_magnitude)
        self.calculate(make_image=False)
        
    @property
    def redshift(self):
        return self._redshift
    
    @redshift.setter
    def redshift(self, new_redshift):
        if self._redshift == new_redshift:
            return
        self._redshift = new_redshift
        self.calculate(make_image=False)
    
    @property
    def zmax(self):
        sed = self.recover('_sed')
        twave = sed.wave * u.Unit(sed.waveunits.name)
        bwave = self.recover('spectrograph.wave')
        return (bwave.max() / twave.min() - 1.0).value
    
    @property
    def zmin(self):
        sed = self.recover('_sed')
        twave = sed.wave * u.Unit(sed.waveunits.name)
        bwave = self.recover('spectrograph.wave')
        return max((bwave.min() / twave.max() - 1.0).value, 0.0)
    
    def calculate(self):
        """
        This should have a means of calculating the exposure time, SNR, 
        and/or limiting magnitude.
        """
        
        raise NotImplementedError
        

class PhotometricExposure(Exposure):
    """
    A subclass of the base Exposure model, for photometric ETC calculations.
    """
    
    def calculate(self, make_image=False, photon_count=0,gain=1000.):
        """
        Wrapper to calculate the exposure time, SNR, or limiting magnitude, 
        based on the other two. The "unknown" attribute controls which of these
        parameters is calculated.
        """
        if self._disable:
            return False
        if self.camera is None or self.telescope is None:
            return False
        status = {'magnitude': self._update_magnitude,
                  'exptime': self._update_exptime,
                  'snr': self._update_snr}[self.unknown]()
        return status
    
    #Calculation methods
    
    @property
    def _fstar(self):
        """
        Calculate the stellar flux as per Eq 2 in the SNR equation paper.
        """
        mag = self.recover('magnitude')
        (f0, c_ap, D, dlam) = self.recover('camera.ab_zeropoint', 
                                           'camera.ap_corr', 
                                           'telescope.effective_aperture', 
                                           'camera.derived_bandpass')
        
        m = 10.**(-0.4*(mag.value))
        D = D.to(u.cm)
        
        fstar = f0 * c_ap * np.pi / 4. * D**2 * dlam * m

        return fstar
    
    
    def _update_exptime(self):
        """
        Calculate the exposure time to achieve the desired S/N for the 
        given SED.
        """
        
        self.camera._print_initcon(self.verbose)
        
        #We no longer need to check the inputs, since they are now tracked
        #attributes instead.
               
        #Convert JsonUnits to Quantities for calculations
        (_snr, _nexp) = self.recover('snr', 'n_exp')
        (_total_qe, _detector_rn, _dark_current) = self.recover('camera.total_qe', 
                'camera.detector_rn', 'camera.dark_current')
        
        snr2 = -(_snr**2)
        fstar = self._fstar
        fsky = self.camera._fsky(verbose=self.verbose)
        Npix = self.camera._sn_box(self.verbose)
        thermal = pre_decode(self.camera.c_thermal(verbose=self.verbose))
        
        a = (_total_qe * fstar)**2
        b = snr2 * (_total_qe * (fstar + fsky) + thermal + _dark_current * Npix)
        c = snr2 * _detector_rn**2 * Npix * _nexp
        
        texp = ((-b + np.sqrt(b**2 - 4*a*c)) / (2*a)).to(u.s)
        
        #serialize with JsonUnit for transportation
        self._exptime = pre_encode(texp)
        
        return True #completed successfully
        
    def _update_magnitude(self):
        """
        Calculate the limiting magnitude given the desired S/N and exposure
        time.
        """
        
        self.camera._print_initcon(self.verbose)
        
        #We no longer need to check the inputs, since they are now tracked
        #attributes instead.
            
        #Grab values for calculation
        (_snr, _exptime, _nexp) = self.recover('snr', 'exptime', 'n_exp')
        (f0, c_ap, D, dlam) = self.recover('camera.ab_zeropoint', 
                                           'camera.ap_corr', 
                                           'telescope.effective_aperture', 
                                           'camera.derived_bandpass')
        (QE, RN, DC) = self.recover('camera.total_qe', 
                                    'camera.detector_rn', 
                                    'camera.dark_current')
        
        exptime = _exptime.to(u.s)
        D = D.to(u.cm)
        fsky = self.camera._fsky(verbose=self.verbose)
        Npix = self.camera._sn_box(self.verbose)
        c_t = pre_decode(self.camera.c_thermal(verbose=self.verbose))
        
        snr2 = -(_snr ** 2)
        a0 = (QE * exptime)**2
        b0 = snr2 * QE * exptime
        c0 = snr2 * ((QE * fsky + c_t + Npix * DC) * exptime + (RN**2 * Npix * _nexp))
        k = (-b0 + np.sqrt(b0**2 - 4. * a0 * c0)) / (2. * a0)
        
        flux = (4. * k) / (f0 * c_ap * np.pi * D**2 * dlam)
        
        self._magnitude = pre_encode(-2.5 * np.log10(flux.value) * u.mag('AB'))
        
        return True #completed successfully
    
    def _update_snr(self):
        """
        Calculate the SNR for the given exposure time and SED.
        """
        
        self.camera._print_initcon(self.verbose)
        
        #We no longer need to check the inputs, since they are now tracked
        #attributes instead.
            
        #Convert JsonUnits to Quantities for calculations
        (_exptime, _nexp, n_bands) = self.recover('_exptime', 'n_exp',
                                                  'camera.n_bands')
        (_total_qe, _detector_rn, _dark_current) = self.recover('camera.total_qe',
                             'camera.detector_rn', 'camera.dark_current')
        
        #calculate everything
        number_of_exposures = np.full(n_bands, _nexp)
        desired_exp_time = (np.full(n_bands, _exptime.value) * _exptime.unit).to(u.second)
        time_per_exposure = desired_exp_time / number_of_exposures
        
        fstar = self._fstar
        signal_counts = _total_qe * fstar * desired_exp_time
        
        fsky = self.camera._fsky(verbose=self.verbose)
        sky_counts = _total_qe * fsky * desired_exp_time
        
        shot_noise_in_signal = np.sqrt(signal_counts)
        shot_noise_in_sky = np.sqrt(sky_counts)
        
        sn_box = self.camera._sn_box(self.verbose)
        
        read_noise = _detector_rn**2 * sn_box * number_of_exposures
        dark_noise = sn_box * _dark_current * desired_exp_time

        thermal = pre_decode(self.camera.c_thermal(verbose=self.verbose))
        
        thermal_counts = desired_exp_time * thermal
        snr = signal_counts / np.sqrt(signal_counts + sky_counts + read_noise
                                      + dark_noise + thermal_counts)
        
        if self.verbose:
            print('# of exposures: {}'.format(_nexp))
            print('Time per exposure: {}'.format(time_per_exposure[0]))
            print('Signal counts: {}'.format(nice_print(signal_counts)))
            print('Signal shot noise: {}'.format(nice_print(shot_noise_in_signal)))
            print('Sky counts: {}'.format(nice_print(sky_counts)))
            print('Sky shot noise: {}'.format(nice_print(shot_noise_in_sky)))
            print('Total read noise: {}'.format(nice_print(read_noise)))
            print('Dark current noise: {}'.format(nice_print(dark_noise)))
            print('Thermal counts: {}'.format(nice_print(thermal_counts)))
            print()
            print('SNR: {}'.format(snr))
            print('Max SNR: {} in {} band'.format(snr.max(), self.camera.bandnames[snr.argmax()]))
        
        #serialize with JsonUnit for transportation
        self._snr = pre_encode(snr)
        
        return True #completed successfully

class SpectrographicExposure(Exposure):
    """
    A subclass of the base Exposure model, for spectroscopic ETC calculations.
    """
    
    def calculate(self, make_image=0,photon_count=0,gain=1000.):
        """
        Wrapper to calculate the exposure time, SNR, or limiting magnitude, 
        based on the other two. The "unknown" attribute controls which of these
        parameters is calculated.
        """
        self.gain_pollux = gain
        self.counting_mode = photon_count
        if self._disable:
            return False
        if self.spectrograph is None or self.telescope is None:
            return False
        
        #At the moment, we only calculate the SNR.
        if self.unknown != "snr":
            raise NotImplementedError("Only SNR calculations currently supported")
        self._update_snr()
        if self.spectrograph.name == 'POLLUX' and make_image == True:
            print('POLLUX')
            self._create_2dmap()
        
    def _update_snr(self):
        """
        Calculate the SNR based on the current SED and spectrograph parameters.
        """
        
        if self.verbose:
            msg1 = "Creating exposure for {} ({})".format(self.telescope.name,
                                                           self.telescope.recover('aperture'))
            msg2 = " with {} in mode {}".format(self.spectrograph.name, self.spectrograph.mode)
            print(msg1 + msg2)
        print('You requested the calculations for: ',self.spectrograph.name, self.spectrograph._mode)
        
        sed, _exptime = self.recover('sed', 'exptime')
        _wave, aeff, bef, aper, R, wrange = self.recover('spectrograph.wave', 
                                                         'spectrograph.aeff', 
                                                         'spectrograph.bef',
                                                         'telescope.aperture',
                                                         'spectrograph.R',
                                                         'spectrograph.wrange')
        exptime = _exptime.to(u.s)[0] #assume that all are the same
        if sed.fluxunits.name == "abmag":
            funit = u.ABmag
        elif sed.fluxunits.name == "photlam":
            funit = u.ph / u.s / u.cm**2 / u.AA
        else:
            funit = u.Unit(sed.fluxunits.name)
        wave = _wave.to(u.AA)
        swave = (sed.wave * u.Unit(sed.waveunits.name)).to(u.AA)
        sflux = (sed.flux * funit).to(u.erg / u.s / u.cm**2 / u.AA, equivalencies=u.spectral_density(swave))
        wave = wave.to(swave.unit)
        delta_lambda = self.recover('spectrograph.delta_lambda').to(u.AA / u.pix)
        iflux = np.interp(wave, swave, sflux, left=0., right=0.) * (u.erg / u.s / u.cm**2 / u.AA)
        phot_energy = const.h.to(u.erg * u.s) * const.c.to(u.cm / u.s) / wave.to(u.cm) / u.ct
        scaled_aeff = aeff * (aper / (15 * u.m))**2
        source_counts = iflux / phot_energy * scaled_aeff * exptime * delta_lambda
        bg_counts = bef / phot_energy * scaled_aeff * exptime * delta_lambda
        if self.spectrograph.name == 'POLLUX':
            count_centr_pix = 0.22 #percentage of counts in central pixel
            box_pixel = 2.4**2
            dark = (box_pixel/3600.)*exptime/u.s * bg_counts.unit # dark noise
            ron = 50./self.gain_pollux # readout noise
            if self.counting_mode == 1:
                num_frame, exp_frame, source_counts = self.compute_num_frame(exptime,source_counts*count_centr_pix,self.gain_pollux)
                cc_n = 0.005 * num_frame * box_pixel * bg_counts.unit # clock induced charge noise
                mask_final = source_counts.value*self.gain_pollux >= num_frame*250.
                source_counts[mask_final] = num_frame*250.* bg_counts.unit
                source_counts[~mask_final] = 0.* bg_counts.unit
                snr = source_counts/np.sqrt(source_counts + 0.75*(bg_counts+dark+ cc_n))
            else:
                enf = 2. #eccess noise factor
                num_frame, exp_frame, source_counts, mask_gain = self.compute_num_frame(exptime,source_counts*count_centr_pix,self.gain_pollux)
                source_counts[mask_gain] = 8.e4* source_counts.unit*num_frame
                if self.gain_pollux > 1:
                     '''Do nothing '''             
                else:
                    ron = 3.1  
                    enf = 1.
                snr = source_counts / np.sqrt((source_counts + bg_counts+dark)*enf+(ron**2*box_pixel*num_frame* bg_counts.unit))
            print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            print('This object required ', num_frame, ' frames of ', exp_frame.value, 's each for a total exposure time of ', exp_frame.value*num_frame, ' s')   
            print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        else:
            snr = source_counts / np.sqrt(source_counts + bg_counts)
        
        
        if self.verbose:
            print("SNR: {}".format(snr))
        
        self._snr = pre_encode(snr)
        
    def _create_2dmap(self):
        """
        Calculate the SNR based on the current SED and spectrograph parameters.
        """
        print('Starting the calculations of POLLUX 2D image')
        if self.verbose:
            msg1 = "Creating exposure for {} ({})".format(self.telescope.name,
                                                           self.telescope.recover('aperture'))
            msg2 = " with {} in mode {}".format(self.spectrograph.name, self.spectrograph.mode)
            print(msg1 + msg2)
        
        sed, _exptime = self.recover('sed', 'exptime')
        _wave, bef, aper, wrange, R = self.recover('spectrograph.wave', 'spectrograph.bef','telescope.aperture','spectrograph.wrange','spectrograph.R')
        self.R = R
        exptime = _exptime.to(u.s)[0] #assume that all are the same
        if sed.fluxunits.name == "abmag":
            funit = u.ABmag
        elif sed.fluxunits.name == "photlam":
            funit = u.ph / u.s / u.cm**2 / u.AA
        else:
            funit = u.Unit(sed.fluxunits.name)
        script_dir = os.path.abspath(os.path.dirname(__file__))
        with open(script_dir+'/../data/POLUX_map.pickle', 'rb') as h:
            maps_pollux = pickle.load(h)
        if self.spectrograph.mode == 'NUV_POL':
            map_pollux = maps_pollux[0][0]
            map_pollux_p = maps_pollux[0][1]
            self.num_pix_x = 20460
            self.num_pix_y = 3846
        elif self.spectrograph.mode == 'MUV_POL':
            map_pollux = maps_pollux[1][0]
            map_pollux_p = maps_pollux[1][1]
            self.num_pix_x = 20460
            self.num_pix_y = 3076
        elif self.spectrograph.mode == 'FUV_POL':
            map_pollux = maps_pollux[2][0]
            map_pollux_p = maps_pollux[2][1]
            self.num_pix_x = 31306
            self.num_pix_y = 3076
        elif self.spectrograph.mode == 'NUV_SPEC':
            map_pollux = maps_pollux[3]
            self.num_pix_x = 20460
            self.num_pix_y = 3846
        elif self.spectrograph.mode == 'MUV_SPEC':
            map_pollux = maps_pollux[4]
            self.num_pix_x = 20460
            self.num_pix_y = 3076
        elif self.spectrograph.mode == 'FUV_SPEC':
            map_pollux = maps_pollux[5]
            self.num_pix_x = 31306
            self.num_pix_y = 3076
        matrix=matrix_mask=wave_mask = np.zeros((self.num_pix_y,self.num_pix_x)) #=background_mask=source_mask
        for i in range(len(map_pollux[2])):
            print('Number of orders computed: ', i)
            matrix_mask,wave_mask1 = self._compute_matrix_pollux(map_pollux,sed,[_wave.to(u.AA),bef],funit,aper,exptime,self.num_pix_x,self.num_pix_y,i) #,background_mask1,source_mask1
            matrix = matrix + matrix_mask
            #background_mask = background_mask+background_mask1
            #source_mask = source_mask + source_mask1
            wave_mask = wave_mask + wave_mask1
            if self.spectrograph.mode == 'NUV_POL' or self.spectrograph.mode == 'MUV_POL' or self.spectrograph.mode == 'FUV_POL':
                matrix_mask,wave_mask1 = self._compute_matrix_pollux(map_pollux_p,sed,[_wave.to(u.AA),bef],funit,aper,exptime,self.num_pix_x,self.num_pix_y,i)#,background_mask1,source_mask1
                matrix = matrix + matrix_mask
                #background_mask = background_mask+background_mask1
                #source_mask = source_mask + source_mask1
                wave_mask = wave_mask + wave_mask1
            
        pixelized_matrix = self.bin_ndarray(matrix, new_shape=(np.int(self.num_pix_y/2),np.int(self.num_pix_x/2)), operation='sum')
        wave_image = self.bin_ndarray(wave_mask, new_shape=(np.int(self.num_pix_y/2),np.int(self.num_pix_x/2)), operation='sum')
        #make ccd image
        
        dim_array = np.shape(pixelized_matrix)#len(pixelized_matrix[0])
        dark = (1/3600.)*exptime/u.s #* bg_counts.unit
        noise_dark = np.ones((dim_array))*dark #dark current image
        
        image_electron = np.random.poisson(lam=pixelized_matrix+noise_dark)    
        print('You requested the counting_mode=%i and the gain=%i' %(self.counting_mode,self.gain_pollux))          
        if self.counting_mode == 1:
            ron_image = 0.
            gain_image = np.random.normal(np.ones((dim_array)),0.001,dim_array) # pixel non uniformity image
            num_frame, exp_frame, image_electron = self.compute_num_frame(exptime,gain_image*0.75*image_electron * u.dimensionless_unscaled,self.gain_pollux)
            cc_n = 0.005 * num_frame
            final_image = image_electron*self.gain_pollux+cc_n*0.75*gain_image
            mask_final = final_image >= num_frame*250.
            final_image[mask_final] = num_frame*250.
            final_image[~mask_final] = 0.
        else:
            if self.gain_pollux > 1.:
                gain_err = np.sqrt(2.)
                max_register = 5.e5
                ron=50./self.gain_pollux
            else:
                gain_err = 0.015
                max_register = 2.8e5
                ron= 3.1
            gain_image = np.random.normal(np.ones((dim_array))*self.gain_pollux,gain_err,dim_array) # pixel non uniformity image with excess noise factor
            num_frame, exp_frame, image_electron,mask_gain = self.compute_num_frame(exptime,image_electron * u.dimensionless_unscaled,self.gain_pollux)
            ron = ron*num_frame
            ron_err = ron*0.015
            ron_image = np.random.normal(ron,ron_err,dim_array) # ron image
            final_image = image_electron*gain_image+ron_image
            mask_gain = final_image/num_frame >= max_register
            final_image[mask_gain] = max_register *num_frame
        
        #final_image = np.zeros((self.num_pix_y,self.num_pix_x))
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('This object required ', num_frame, ' frames of ', exp_frame.value, 's each')   
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        
        self._final_image = pre_encode(final_image)
        self._wave_image = pre_encode(wave_image)
        print('Image done!')
        
    def bin_ndarray(self,ndarray, new_shape, operation='sum'):
        """
        Bins an ndarray in all axes based on the target shape, by summing or
        averaging.

        Number of output dimensions must match number of input dimensions and 
        new axes must divide old ones.
        """
        operation = operation.lower()
        if not operation in ['sum', 'mean']:
            raise ValueError("Operation not supported.")
        if ndarray.ndim != len(new_shape):
            raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,new_shape))
        compression_pairs = [(d, c//d) for d,c in zip(new_shape, ndarray.shape)]
        flattened = [l for p in compression_pairs for l in p]
        ndarray = ndarray.reshape(flattened)
        for i in range(len(new_shape)):
            op = getattr(ndarray, operation)
            ndarray = op(-1*(i+1))
        return ndarray


    def _compute_matrix_pollux(self,map_p,sed,bef,funit,aper,exptime,num_pix_x,num_pix_y,i):
       
        wave = map_p[2][i]
        bwave = bef[0] #background 
        bflux = bef[1] 
        swave = (sed.wave * u.Unit(sed.waveunits.name)).to(u.AA) #signal
        sflux = (sed.flux * funit).to(u.erg / u.s / u.cm**2 / u.AA, equivalencies=u.spectral_density(swave))
        aeff = map_p[3][i]*0.87
        delta_x = map_p[4][i]
        index_y = map_p[0][i]
        index_x = map_p[1][i]
        iflux = np.interp(wave, swave, sflux, left=0., right=0.) * (u.erg / u.s / u.cm**2 / u.AA)
        bef_flux = np.interp(wave, bwave, bflux, left=0., right=0.)
        delta_lambda = wave/self.R
        #print(np.shape(iflux))
        phot_energy = const.h.to(u.erg * u.s) * const.c.to(u.cm / u.s) / (wave*1.e-8) / u.ct
        scaled_aeff = aeff * (aper / (15 * u.m))**2
        source_counts = iflux / phot_energy * scaled_aeff * exptime * delta_lambda
        bg_counts = bef_flux / phot_energy * scaled_aeff * exptime * delta_lambda
        matrix_mask = np.zeros((num_pix_y,num_pix_x))
        background_mask = np.zeros((num_pix_y,num_pix_x))
        matrix_mask[index_y.astype(int),index_x.astype(int)] = source_counts
        #source_mask = np.copy(matrix_mask)
        wave_mask = np.zeros((num_pix_y,num_pix_x))
        wave_mask[index_y.astype(int),index_x.astype(int)] = wave
        background_mask[index_y.astype(int)-2, index_x.astype(int)] = bg_counts 
        background_mask[index_y.astype(int)-1, index_x.astype(int)] = bg_counts 
        background_mask[index_y.astype(int), index_x.astype(int)] = bg_counts 
        background_mask[index_y.astype(int)+1, index_x.astype(int)] = bg_counts 
        background_mask[index_y.astype(int)+2, index_x.astype(int)] = bg_counts 
        max_in_x = np.max(index_x.astype(int))
        min_in_x = np.min(index_x.astype(int))
        max_in_y = np.max(index_y.astype(int))
        min_in_y = np.min(index_y.astype(int))
        matrix_mask = matrix_mask+background_mask
        new_matrix = matrix_mask[min_in_y-20:max_in_y+20,:]
        convolved_matrix = gaussian_filter(new_matrix, sigma=delta_x)
        matrix_mask[min_in_y-20:max_in_y+20,:] = convolved_matrix
        
        return matrix_mask,wave_mask#,background_mask,source_mask



    def compute_num_frame(self,exptime,signal,gain):
        """
        """
        
        fwc = 8.e4 # full well capacity
        if gain > 1.:
            register_c = 5.e5 #register capacity
            min_exp_t = 0.120 # min exposure time
        else:
            register_c = 2.8e5 
            min_exp_t = 24.
        #cc_n = 0.005 # clock charge induced noise
        num_frame = 1
        exp_frame = exptime
        count_centr_pix = 0.22
        mask_source2 = (signal.value>=np.min(signal.value))
        if self.counting_mode == 1:
            threshold = 250.*signal.unit
            signal = 0.75*signal*gain
            mask_threshold = (np.max(signal.value) > threshold.value)*(np.max(signal.value) <= threshold.value+5)
            if signal[mask_threshold].size >0:
                '''do nothing'''
            else:
                exp_frame = exptime*threshold/np.max(signal)
                num_frame = int(math.ceil(exptime/exp_frame))
                if exp_frame.value < min_exp_t :
                    exp_frame = min_exp_t*exptime.unit
                    num_frame = int(math.ceil(exptime/exp_frame))
                mask_register = (signal/num_frame)>threshold#register_c.value
                signal[mask_register] = threshold*num_frame
                if signal[mask_register].size >0:
                    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                    print('WARNING: the signal level might be saturated in some region of the spectrum')
                    print('You can use a higher magnitude/lower exposure time to avoid it')
                    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            return num_frame, exp_frame, signal/gain
        else:
            mask_source = (signal.value>=fwc)
            if signal[mask_source].size > 0:
                exp_frame = exptime*fwc* signal.unit/np.max(signal)
                if exp_frame.value < min_exp_t :
                    exp_frame = min_exp_t*exptime.unit
                    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                    print('WARNING: the signal level is saturated in some region of the spectrum')
                    print('You can use a higher magnitude/lower exposure time to avoid it')
                    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                num_frame = int(math.ceil(exptime/exp_frame))
                mask_source = ((signal.value/num_frame)>fwc)
                mask_source2 = ((signal.value/num_frame)<=fwc)
                signal[mask_source] = fwc* signal.unit*num_frame
            mask_gain = ((signal.value*gain/num_frame)>=register_c)
            if signal[mask_gain].size > 0:
                exp_frame = exp_frame*register_c* signal.unit/np.max(signal*gain/num_frame)
                if exp_frame.value < min_exp_t :
                    exp_frame = min_exp_t*exptime.unit
                    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                    print('WARNING: the signal level is saturated in some region of the spectrum')
                    print('You can use a higher magnitude/lower exposure time to avoid it')
                    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                num_frame = int(math.ceil(exptime/(exp_frame*num_frame)))
                mask_source = ((signal.value/num_frame)>fwc)
                mask_source2 = ((signal.value/num_frame)<=fwc)
                mask_gain = ((signal.value*gain/num_frame)>register_c)
                signal[mask_source] = fwc* signal.unit*num_frame
                signal[~mask_source] = signal[~mask_source]/count_centr_pix
            #signal[mask_source2] = signal/count_centr_pix
            return num_frame, exp_frame, signal, mask_gain
          
