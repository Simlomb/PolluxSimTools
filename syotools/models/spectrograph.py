#!/usr/bin/env python
"""
Created on Sat Oct 15 16:56:40 2016

@author: gkanarek, tumlinson
"""

from __future__ import (print_function, division, absolute_import, with_statement,
                        nested_scopes, generators)

import numpy as np
import pickle
import astropy.units as u
from astropy.table import QTable

from syotools.models.base import PersistentModel
from syotools.models.exposure import SpectrographicExposure
from syotools.defaults import default_spectrograph
from syotools.defaults import default_spectropolarimeter
from syotools.utils import pre_encode

class Spectrograph(PersistentModel):
    """
    The basic spectrograph class, which provides parameter storage for 
    optimization.
    
    Attributes: #adapted from the original in Telescope.py
        telescope    - the Telescope object associated with this spectrograph
        exposures    - the list of Exposures taken with this spectrograph
    
        name         - name of the spectrograph (string)
        
        modes        - supported observing modes (list)
        descriptions - description of supported observing modes (dict)
        mode         - current observing mode (string)
        bef          - background emission function in erg/s/cm3/res_element (float array)
        R            - spectral resolution (float)
        wrange        - effective wavelength range (2-element float array)
        wave         - wavelength in Angstroms (float array)
        aeff         - effective area at given wavelengths in cm^2 (float array)
        
        _lumos_default_file - file path to the fits file containing LUMOS values
        
        _default_model - used by PersistentModel
    """
    
    _default_model = default_spectrograph
    
    telescope = None
    exposures = []
    
    _lumos_default_file = ''
    
    name = ''
    modes = []
    descriptions = {}
    bef = pre_encode(np.zeros(0, dtype=float) * (u.erg / u.cm**2 / u.s / u.AA))
    R = pre_encode(0. * u.dimensionless_unscaled) 
    wave = pre_encode(np.zeros(0, dtype=float) * u.AA)
    aeff = pre_encode(np.zeros(0, dtype=float) * u.cm**2)
    wrange = pre_encode(np.zeros(2, dtype=float) * u.AA)
    _mode = ''
    
    #Property wrapper for mode, so that we can use a custom setter to propagate
    #mode updates to all the rest of the parameters
    
    @property
    def mode(self):
        return self._mode
    
    @mode.setter
    def mode(self, new_mode):
        """
        Mode is used to set all the other parameters
        """
        
        nmode = new_mode.upper()
        if self._mode == nmode or nmode not in self.modes:
            return
        self._mode = nmode
        table = QTable.read(self._lumos_default_file, nmode)
                
        self.R = pre_encode(table.meta['R'] * u.pix)
        self.wave = pre_encode(table['Wavelength'])
        self.bef = pre_encode(table['BEF'] / self.recover('delta_lambda'))
        self.aeff = pre_encode(table['A_Eff'])
        wrange = np.array([table.meta['WAVE_LO'], table.meta['WAVE_HI']]) * u.AA
        self.wrange = pre_encode(wrange)

    @property
    def delta_lambda(self):
        wave, R = self.recover('wave', 'R')
        return pre_encode(wave / R)
    
    def create_exposure(self):
        new_exposure = SpectrographicExposure()
        self.add_exposure(new_exposure)
        return new_exposure
    
    def add_exposure(self, exposure):
        self.exposures.append(exposure)
        exposure.spectrograph = self
        exposure.telescope = self.telescope
        exposure.calculate(make_image=False)

class Spectropolarimeter(Spectrograph):
    """
    The basic spectropolarimeter class for POLLUX, which provides parameter storage for 
    optimization.
    
    Attributes: #adapted from the original in Telescope.py
        telescope    - the Telescope object associated with this spectrograph
        exposures    - the list of Exposures taken with this spectrograph
    
        name         - name of the spectrograph (string)
        
        modes        - supported observing modes (list)
        descriptions - description of supported observing modes (dict)
        mode         - current observing mode (string)
        bef          - background emission function in erg/s/cm3/res_element (float array)
        R            - spectral resolution (float)
        wrange        - effective wavelength range (2-element float array)
        wave         - wavelength in Angstroms (float array)
        aeff         - effective area at given wavelengths in cm^2 (float array)
        
        _lumos_default_file - file path to the fits file containing LUMOS values
        
        _default_model - used by PersistentModel
    """
    
    _default_model = default_spectropolarimeter

'''
class SpectropolarimeterImage(Spectrograph):
    
    _default_model = default_spectropolarimeterimage
    
    telescope = None
    exposures = []
    
    _lumos_default_file = ''
    
    name = ''
    modes = []
    descriptions = {}
    #bef = pre_encode(np.zeros(0, dtype=float) * (u.erg / u.cm**2 / u.s / u.AA))
    R = pre_encode(0. * u.dimensionless_unscaled) 
    wave = pre_encode(np.zeros(0,0 dtype=float) * u.AA)
    index_y = pre_encode(np.zeros(0,0 dtype=float) * u.dimensionless_unscaled)
    index_y = pre_encode(np.zeros(0,0 dtype=float) * u.dimensionless_unscaled)
    aeff = pre_encode(np.zeros(0,0 dtype=float) * u.cm**2)
    wrange = pre_encode(np.zeros(2, dtype=float) * u.AA)
    _mode = ''

    @mode.setter
    def mode(self, new_mode):
        """
        Mode is used to set all the other parameters
        """
        
        nmode = new_mode.upper()
        if self._mode == nmode or nmode not in self.modes:
            return
        self._mode = nmode
        with open(self._lumos_default_file, 'rb') as handle:
            b = pickle.load(handle)

        table = b[nmode]
        if nmode<=2:
            self.Rp = pre_encode(table[0][4] * u.pix)
            self.wavep = pre_encode(table[0][2])
            self.aeffp = pre_encode(table[0][3])
            wrangep = np.array([np.min(table[0][2]), np.max(table[0][2])]) * u.AA
            self.wrangep = pre_encode(wrangep)
            self.index_yp = pre_encode(table[0][0] * u.pix)
            self.index_xp = pre_encode(table[0][1] * u.pix)
            self.Rs = pre_encode(table[1][4] * u.pix)
            self.waves = pre_encode(table[1][2])
            self.aeffs = pre_encode(table[1][3])
            wranges = np.array([np.min(table[1][2]), np.max(table[1][2])]) * u.AA
            self.wranges = pre_encode(wranges)
            self.index_ys = pre_encode(table[1][0] * u.pix)
            self.index_xs = pre_encode(table[1][1] * u.pix)
        else:
            self.R = pre_encode(table[4] * u.pix)
            self.wave = pre_encode(table[2])
            self.aeff = pre_encode(table[3])
            wrange = np.array([np.min(table[2]), np.max(table[2])]) * u.AA
            self.wrange = pre_encode(wrange)
            self.index_y = pre_encode(table[0] * u.pix)
            self.index_x = pre_encode(table[1] * u.pix)
'''
