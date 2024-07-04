"""
A minimal module to weight neutrinos with some default flux and oscillation parameters.

This can be used to produce reference weights in case those are not present in
HDF5 files.
"""

import os
import numpy as np
from pisa.utils.flux_weights import load_2d_table, calculate_2d_flux_weights
from pisa.stages.osc.osc_params import OscParams
from pisa.stages.osc.decay_params import DecayParams
from pisa.stages.osc.layers import Layers
from pisa.stages.osc.prob3numba.numba_osc_hostfuncs import propagate_array, fill_probs
from pisa.utils.resources import find_resource


class GENIENeutrinoWeighter(object):
    """
    A class for adding weights to neutrino events that take into account flux and
    oscillations. This assumes that the events were generated with GENIE.
    """

    def __init__(self, nufit_version="2.2", neutrino_decay=False):

        self.flux_table = load_2d_table("flux/honda-2015-spl-solmin-aa.d")

        self.earth_model = os.path.expandvars(
            "$I3_SRC/prob3/resources/oscillations/PREM_10layer.dat"
        )
        self.detector_depth = 2.0
        self.prop_height = 20.0

        

        # setup the layers
        # if self.params.earth_model.value is not None:
        earth_model = find_resource("osc/PREM_12layer.dat")

        YeI = 0.4656
        YeM = 0.4957
        YeO = 0.4656

        # height
        detector_depth = 2.0  # in km
        prop_height = 20.0  # in km

        self.layers = Layers(earth_model, detector_depth, prop_height)
        self.layers.setElecFrac(YeI, YeO, YeM)
        
        self.osc_params = OscParams()
        self.decay_flag = None
        self.decay_params = None
        if neutrino_decay:
            self.decay_flag = 1
            self.decay_params = DecayParams()
            self.set_decay_param()
        else :
            self.decay_flag = -1
      
        self.set_nufit(nufit_version)
        
    def set_decay_param(self):
        self.decay_params.decay_alpha3 = 1e-4
    
    def set_nufit(self, version):
        assert version in ["2.2", "4.0"]
        if version == "2.2":
            # values from NuFIT 2.2 (2016) to be directly comparable to
            # those used to calculate oscNext reference weights
            self.osc_params.theta12 = 33.72 * np.pi / 180
            self.osc_params.theta13 = 8.46 * np.pi / 180
            self.osc_params.theta23 = 41.5 * np.pi / 180
            self.osc_params.dm21 = 7.49e-5
            self.osc_params.dm31 = 2.526e-3
            self.osc_params.deltacp = 289.0 * np.pi / 180
        elif version == "4.0":
            # http://www.nu-fit.org/?q=node/177
            self.osc_params.theta12 = 33.82 * np.pi / 180
            self.osc_params.theta13 = 8.61 * np.pi / 180
            self.osc_params.theta23 = 49.6 * np.pi / 180
            self.osc_params.dm21 = 7.39e-5
            self.osc_params.dm31 = 2.525e-3
            self.osc_params.deltacp = 215.0 * np.pi / 180
        elif version == "4.1":
            # http://www.nu-fit.org/?q=node/211
            self.osc_params.theta12 = 33.82 * np.pi / 180
            self.osc_params.theta13 = 8.61 * np.pi / 180
            self.osc_params.theta23 = 48.3 * np.pi / 180
            self.osc_params.dm21 = 7.49e-5
            self.osc_params.dm31 = 2.523e-3
            self.osc_params.deltacp = 222.0 * np.pi / 180
                
    def calc_probs(self, nubar, e_array, rho_array, len_array, out):
        """wrapper to execute osc. calc"""

        mix_matrix = self.osc_params.mix_matrix_complex
        # the matter potential matrix part of the Hamiltonian in the flavor basis
        std_mat_pot_matrix = np.zeros((3, 3), dtype=complex)
        # in the 3-flavor standard oscillation case, the diagonal component of the
        # matter potential can be subtracted and only the entry [0, 0]
        # is non-zero (for nu_e CC interactions).
        std_mat_pot_matrix[0, 0] += 1.0
        
        if (self.decay_flag==1):
            mat_decay = self.decay_params.decay_matrix
        else:
            mat_decay = np.zeros((3, 3), dtype=complex)
            
        propagate_array(
            self.osc_params.dm_matrix,
            mix_matrix,
            std_mat_pot_matrix,
            self.decay_flag,
            mat_decay,
            nubar,
            e_array,
            rho_array,
            len_array,
            out=out,
        )

    def __call__(self, energy, cos_zenith, nubar, weighted_aeff):

        #### Calculate flux
        out_names = ["nu_flux_nominal"] * 2 + ["nubar_flux_nominal"] * 2
        indices = [0, 1, 0, 1]
        tables = ["nue", "numu", "nuebar", "numubar"]

        nu_flux_nominal = np.zeros((len(energy), 2))
        nubar_flux_nominal = np.zeros((len(energy), 2))

        flux_data = {
            "nu_flux_nominal": nu_flux_nominal,
            "nubar_flux_nominal": nubar_flux_nominal,
        }

        for out_name, index, table in zip(out_names, indices, tables):
            print(f"Calculating nominal {table} flux")
            calculate_2d_flux_weights(
                true_energies=energy,
                true_coszens=cos_zenith,
                en_splines=self.flux_table[table],
                out=flux_data[out_name][:, index],
            )

        #### Calculate oscillation probabilities
        self.layers.calcLayers(cos_zenith)
        densities = self.layers.density.reshape((len(energy), self.layers.max_layers))
        distances = self.layers.distance.reshape((len(energy), self.layers.max_layers))

        probability = np.empty((len(energy), 3, 3))

        print("Calculating oscillation probabilities...")
        self.calc_probs(
            nubar,
            energy,
            densities,
            distances,
            out=probability,
        )

        index = np.arange(len(probability))
        # intuitively I would have thought that this should work with
        # probability[:, 0, flav], but it doesn't. This below is super hacky, but
        # I gave up figuring out how to use Ellipsis correctly here.
        prob_from_nue = probability[index, 0, flav]
        prob_from_numu = probability[index, 1, flav]
        
        # Apply flux and oscillations
        nue_flux = np.where(nubar > 0, nu_flux_nominal[:, 0], nubar_flux_nominal[:, 0])
        numu_flux = np.where(nubar > 0, nu_flux_nominal[:, 1], nubar_flux_nominal[:, 1])

        weight_with_osc = weighted_aeff * np.array([nue_flux * prob_from_nue,
                                                    numu_flux * prob_from_numu])

        return weight_with_osc
        #return probability

class GENIENeutrinoWeighter_Backup(object):
    """
    A class for adding weights to neutrino events that take into account flux and
    oscillations. This assumes that the events were generated with GENIE.
    """

    def __init__(self, nufit_version="2.2", neutrino_decay=False):

        self.flux_table = load_2d_table("flux/honda-2015-spl-solmin-aa.d")

        self.earth_model = os.path.expandvars(
            "$I3_SRC/prob3/resources/oscillations/PREM_10layer.dat"
        )
        self.detector_depth = 2.0
        self.prop_height = 20.0

        

        # setup the layers
        # if self.params.earth_model.value is not None:
        earth_model = find_resource("osc/PREM_12layer.dat")

        YeI = 0.4656
        YeM = 0.4957
        YeO = 0.4656

        # height
        detector_depth = 2.0  # in km
        prop_height = 20.0  # in km

        self.layers = Layers(earth_model, detector_depth, prop_height)
        self.layers.setElecFrac(YeI, YeO, YeM)
        
        self.osc_params = OscParams()
        self.decay_flag = None
        self.decay_params = None
        if neutrino_decay:
            self.decay_flag = 1
            self.decay_params = DecayParams()
            self.set_decay_param()
        else :
            self.decay_flag = -1
      
        self.set_nufit(nufit_version)
        
    def set_decay_param(self):
        self.decay_params.decay_alpha3 = 1e-4
    
    def set_nufit(self, version):
        assert version in ["2.2", "4.0"]
        if version == "2.2":
            # values from NuFIT 2.2 (2016) to be directly comparable to
            # those used to calculate oscNext reference weights
            self.osc_params.theta12 = 33.72 * np.pi / 180
            self.osc_params.theta13 = 8.46 * np.pi / 180
            self.osc_params.theta23 = 41.5 * np.pi / 180
            self.osc_params.dm21 = 7.49e-5
            self.osc_params.dm31 = 2.526e-3
            self.osc_params.deltacp = 289.0 * np.pi / 180
        elif version == "4.0":
            # http://www.nu-fit.org/?q=node/177
            self.osc_params.theta12 = 33.82 * np.pi / 180
            self.osc_params.theta13 = 8.61 * np.pi / 180
            self.osc_params.theta23 = 49.6 * np.pi / 180
            self.osc_params.dm21 = 7.39e-5
            self.osc_params.dm31 = 2.525e-3
            self.osc_params.deltacp = 215.0 * np.pi / 180
        elif version == "4.1":
            # http://www.nu-fit.org/?q=node/211
            self.osc_params.theta12 = 33.82 * np.pi / 180
            self.osc_params.theta13 = 8.61 * np.pi / 180
            self.osc_params.theta23 = 48.3 * np.pi / 180
            self.osc_params.dm21 = 7.49e-5
            self.osc_params.dm31 = 2.523e-3
            self.osc_params.deltacp = 222.0 * np.pi / 180
                
    def calc_probs(self, nubar, e_array, rho_array, len_array, out):
        """wrapper to execute osc. calc"""

        mix_matrix = self.osc_params.mix_matrix_complex
        # the matter potential matrix part of the Hamiltonian in the flavor basis
        std_mat_pot_matrix = np.zeros((3, 3), dtype=complex)
        # in the 3-flavor standard oscillation case, the diagonal component of the
        # matter potential can be subtracted and only the entry [0, 0]
        # is non-zero (for nu_e CC interactions).
        std_mat_pot_matrix[0, 0] += 1.0
        
        if (self.decay_flag==1):
            mat_decay = self.decay_params.decay_matrix
        else:
            mat_decay = np.zeros((3, 3), dtype=complex)
            
        propagate_array(
            self.osc_params.dm_matrix,
            mix_matrix,
            std_mat_pot_matrix,
            self.decay_flag,
            mat_decay,
            nubar,
            e_array,
            rho_array,
            len_array,
            out=out,
        )

    def __call__(self, energy, cos_zenith, flav, nubar, weighted_aeff):

        #### Calculate flux
        out_names = ["nu_flux_nominal"] * 2 + ["nubar_flux_nominal"] * 2
        indices = [0, 1, 0, 1]
        tables = ["nue", "numu", "nuebar", "numubar"]

        nu_flux_nominal = np.zeros((len(energy), 2))
        nubar_flux_nominal = np.zeros((len(energy), 2))

        flux_data = {
            "nu_flux_nominal": nu_flux_nominal,
            "nubar_flux_nominal": nubar_flux_nominal,
        }

        for out_name, index, table in zip(out_names, indices, tables):
            print(f"Calculating nominal {table} flux")
            calculate_2d_flux_weights(
                true_energies=energy,
                true_coszens=cos_zenith,
                en_splines=self.flux_table[table],
                out=flux_data[out_name][:, index],
            )

        #### Calculate oscillation probabilities
        self.layers.calcLayers(cos_zenith)
        densities = self.layers.density.reshape((len(energy), self.layers.max_layers))
        distances = self.layers.distance.reshape((len(energy), self.layers.max_layers))

        probability = np.empty((len(energy), 3, 3))

        print("Calculating oscillation probabilities...")
        self.calc_probs(
            nubar,
            energy,
            densities,
            distances,
            out=probability,
        )

        index = np.arange(len(probability))
        # intuitively I would have thought that this should work with
        # probability[:, 0, flav], but it doesn't. This below is super hacky, but
        # I gave up figuring out how to use Ellipsis correctly here.
        prob_from_nue = probability[index, 0, flav]
        prob_from_numu = probability[index, 1, flav]

        # Apply flux and oscillations
        nue_flux = np.where(nubar > 0, nu_flux_nominal[:, 0], nubar_flux_nominal[:, 0])
        numu_flux = np.where(nubar > 0, nu_flux_nominal[:, 1], nubar_flux_nominal[:, 1])

        weight_with_osc = weighted_aeff * (
            (nue_flux * prob_from_nue) + (numu_flux * prob_from_numu)
        )

        return weight_with_osc
        #return probability
