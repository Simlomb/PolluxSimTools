#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 13:13:28 2017

@author: gkanarek
"""

from __future__ import (print_function, division, absolute_import, with_statement,
                        nested_scopes, generators)
import optparse
import os

script_dir = os.path.abspath(os.path.dirname(__file__))

import numpy as np
import astropy.units as u
from astropy.table import Table

from syotools import cdbs

from syotools.models import Telescope, Spectropolarimeter, SpectrographicExposure as Exposure
from syotools.interface import SYOTool
from syotools.spectra import SpectralLibrary
from syotools.utils import pre_encode, pre_decode
import matplotlib.pyplot             as plt


#We're going to just use the default values for LUVOIR
#establish simtools dir
if 'LUVOIR_SIMTOOLS_DIR' not in os.environ:
    fdir = os.path.abspath(__file__)
    basedir = os.path.abspath(os.path.join(fdir, '..'))
    os.environ['LUVOIR_SIMTOOLS_DIR'] = basedir
    print('eeeee  ', basedir)

if __name__ == '__main__':
    ##########################################################################
    #######################  OPTION ##########################################
    ##########################################################################
    parser = optparse.OptionParser()
   
    parser.add_option("-e", "--exposure_time",
                      help="exposure time in hours", default='1.')
    parser.add_option("-i", "--input_sed",
                      help="Name of the input sed. Possible templates: \n"
                           "o5v  -->  O5V Star, " 
                           "hr1886  -->  B1V Star, "
                           "alplyr  -->  A0V Star, "
                           "alpcmi  -->  FI5V-V, "
                           "g2v  -->  G2V Star, "
                           "gamdra  -->  K5III, "
                           "mdwarf  -->  M1 Dwarf, "
                           "mdwarf2  -->  M3 Dwarf, "
                           "ctts2  -->  Classical TTauri, "
                           "g191b2b  -->  G191B2B (WD), " 
                           "gd71  -->  GD71 (WD), "
                           "gd153  -->  GD153 (WD), "
                           "qso  -->  QSO, "
                           "s99  -->  10 Myr Starburst, "
                           "orion  -->  Orion Nebula, "
                           "nodust  -->  Starburst, No Dust, "
                           "ebv6  -->  Starburst, E(B-V)=0.6, "
                           "syfrt1  -->  Seyfert 1, "
                           "syfrt2  -->  Seyfert 2, "
                           "liner  -->  Liner, "
                           "flam  -->  Flat in F_Lambda", default='qso')
    parser.add_option("-r", "--redshift",
                      help="redshift", default='0.')
    parser.add_option("-m", "--ABmag",
                      help="AB magnitude of the object", default='21.')
    parser.add_option("-a", "--aperture",
                      help="Telescope aperture in m", default='15.')
    parser.add_option("-c", "--channel_mode",
                      help="Channel and operation mode", default='NUV_POL')
    parser.add_option("-s", "--show_plot",
                      help="show the image and save it", default='1')
    parser.add_option("-f", "--file_name",
                      help="Name of the file to be saved", default='test')
    parser.add_option("-n", "--snr",
                      help="Compute the S/N and plot it", default='1')
    parser.add_option("-d", "--full_image",
                      help="Compute the full image and plot it", default='0')
    parser.add_option("-p", "--photon_counting",
                      help="Set the EMCCD in photon counting mode", default='0')
    
    parser.add_option("-g", "--EM_gain",
                      help="Set the EMCCD multiplication gain with numbers from 20 to 1000 (only implemented for the imaging mode) ", default='1000')

    #usage: python main_pollux.py -e 1 -i hr1886 -m 22 -c MUV_SPEC 
    opts, args = parser.parse_args()
    #print "Loading data ..."
    expt_pollux = pre_encode(float(opts.exposure_time) * u.hour)
    ap_pollux = pre_encode(float(opts.aperture) * u.m)
    red_pollux = pre_encode(float(opts.redshift) * u.dimensionless_unscaled)
    mag_pollux = pre_encode(float(opts.ABmag) * u.mag('AB'))
    grating_pollux = str(opts.channel_mode)
    spectrum_pollux = str(opts.input_sed)
    snr_pollux = int(opts.snr)
    image_pollux = int(opts.full_image)
    photon_count = int(opts.photon_counting)
    gain = float(opts.EM_gain)
    show_plot = int(opts.show_plot)
    file_name = str(opts.file_name)

class POLLUX_IMAGE(SYOTool):
    
    tool_prefix = "pollux"
        
    save_models = ["telescope", "camera", "spectrograph", "spectropolarimeter", "exposure"]
    save_params = {"redshift": None, #slider value
                   "renorm_magnitude": None, #slider value
                   "exptime": None, #slider value
                   "grating": ("spectropolarimeter", "mode"), #drop-down selection
                   "aperture": ("telescope", "aperture"), #slider value
                   "spectrum_type": ("exposure", "sed_id"), #drop-down selection
                   "user_prefix": None}
    
    save_dir = os.path.join(os.environ['LUVOIR_SIMTOOLS_DIR'],'saves')
    
    #must include this to set defaults before the interface is constructed
    tool_defaults = {'redshift': pre_encode(0.0 * u.dimensionless_unscaled),
                     'renorm_magnitude': pre_encode(21.0 * u.mag('AB')),
                     'exptime': pre_encode(1.0 * u.hour),
                     'grating': "NUV_POL",
                     'aperture': pre_encode(15.0 * u.m),
                     'spectrum_type': 'qso'}
    def __init__(self):
        print('Initializing the parameters ...')
        
        self.grating = grating_pollux
        self.aperture = ap_pollux
        self.exptime = expt_pollux
        self.redshift = red_pollux 
        self.renorm_magnitude = mag_pollux 
        self.spectrum_type = spectrum_pollux
        self.tool_preinit()
        self.photon_count = photon_count
        self.EM_gain = gain
        self.show_plot = show_plot
        self.file_name = file_name
        if snr_pollux == 1:
            self.make_image = False
            self.update_exposure()
            self.plot_snr()
            print('Saving the file at', basedir,'/pollux_tool/files/')
            np.savetxt(basedir+'/pollux_tool/files/'+self.file_name+'_'+self.grating+'_snr_data.txt', np.array((self.background_wave, self._snr)).T)
        if image_pollux == 1:
            self.make_image = True
            self.update_exposure()
            self.plot_image()
            print('Saving the file at', basedir,'/pollux_tool/files/')
            np.savetxt(basedir+'/pollux_tool/files/'+self.file_name+'_'+self.grating+'_2dimage_data.txt', self._final_image)
        

    def tool_preinit(self):
        """
        Pre-initialize any required attributes for the interface.
        """
        #initialize engine objects
        self.telescope = Telescope(temperature=pre_encode(280.0*u.K))
        self.spectrograph = Spectropolarimeter()
        self.exposure = Exposure()
        self.telescope.add_spectrograph(self.spectrograph)
        self.spectrograph.add_exposure(self.exposure)
        #set interface variables
       
     
    tool_postinit = None
        
    
    def update_exposure(self):
        """
        Update the exposure's parameters and recalculate everything.
        """
        #We turn off calculation at the beginning so we can update everything 
        #at once without recalculating every time we change something
              
        #Update all the parameters
        self.telescope.aperture = self.aperture
        self.spectrograph.mode = self.grating
        self.exposure.exptime = pre_decode(self.exptime)
        self.exposure.redshift = pre_decode(self.redshift)
        self.exposure.sed_id = self.spectrum_type
        self.exposure.renorm_sed(pre_decode(self.renorm_magnitude), 
                                 bandpass='galex,fuv')
        #Now we turn calculations back on and recalculate
        self.exposure.enable(make_image=self.make_image,photon_count=self.photon_count,gain=self.EM_gain)
        #Set the spectrum template
        self.spectrum_template = pre_decode(self.exposure.sed)
        

    def plot_snr(self):
       """
       """
       plt.figure(1)
       plt.subplot(211)
       plt.plot(self.template_wave,self.template_flux, '-b', label='source')
       plt.plot(self.background_wave,self.background_flux, '--k', label='background')
       plt.legend(loc='upper right', fontsize='large',frameon=False)
       plt.xlim(900,4000.)
       #plt.xlabel('Wavelangth [A]', fontsize='x-large')
       plt.ylabel('Flux [erg/s/cm^2/A]', fontsize='x-large')

       plt.subplot(212)
       plt.plot(self.background_wave, self._snr, '-b')
       plt.xlim(900,4000.)
       plt.xlabel('Wavelangth [A]', fontsize='x-large')
       plt.ylabel('S/N per resel', fontsize='x-large')
       if show_plot == 1:
           print('Saving the image at', basedir,'/pollux_tool/plots/')
           plt.savefig(basedir+'/pollux_tool/plots/'+self.file_name+'_'+self.grating+'_snr_plot.png', dpi=300)
           plt.show()
    
    def plot_image(self):
       """
       """
       plt.imshow(self._final_image,  origin='lower left', cmap='hot', interpolation='none',aspect='auto')
       plt.colorbar()
       if show_plot == 1:
           print('Saving the image at', basedir,'/pollux_tool/plots/')
           plt.savefig(basedir+'/pollux_tool/plots/'+self.file_name+'_'+self.grating+'_2dimage_plot.png', dpi=300)
           plt.show()
    
    @property
    def template_wave(self):
        """
        Easy SED wavelength access for the Bokeh widgets.
        """
        return self.exposure.recover('sed').wave
    
    @property
    def template_flux(self):
        """
        Easy SED flux access for the Bokeh widgets.
        """
        sed = self.exposure.recover('sed')
        wave = sed.wave * u.Unit(sed.waveunits.name)
        if sed.fluxunits.name == "abmag":
            funit = u.ABmag
        elif sed.fluxunits.name == "photlam":
            funit = u.ph / u.s / u.cm**2 / u.AA
        else:
            funit = u.Unit(sed.fluxunits.name)
        flux = (sed.flux * funit).to(u.erg / u.s / u.cm**2 / u.AA, 
                equivalencies=u.spectral_density(wave))
        return flux.value
    
    @property
    def background_wave(self):
        """
        Easy instrument wavelength access for the Bokeh widgets.
        """
        bwave = self.spectrograph.recover('wave').to(u.AA)
        return bwave.value
    
    @property
    def background_flux(self):
        """
        Easy instrument background flux access for the Bokeh widgets.
        """
        bef = self.spectrograph.recover('bef').to(u.erg / u.s / u.cm**2 / u.AA)
        return bef.value
    
    @property
    def _snr(self):
        return np.nan_to_num(self.exposure.recover('snr').value)

    @property
    def _final_image(self):
        return np.nan_to_num(self.exposure.recover('final_image'))
    
    
POLLUX_IMAGE()
