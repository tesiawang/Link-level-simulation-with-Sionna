import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
from pathlib import Path
import argparse
import sys
import pickle  

tf.get_logger().setLevel('ERROR')

# set root path
root_path = str(Path(__file__).parent)
sys.path.append(root_path)
from args import BasicConfig
import sionna
from sionna.ofdm.channel_estimation import BaseChannelEstimator, LSChannelEstimator, LMMSEInterpolator



# ----------------------------- ARGUMENT CONFIG ----------------------------- #
parser = argparse.ArgumentParser()
# path config
parser.add_argument("--num_bs_ant", type=int, default=4, required=False)
parser.add_argument("--fft_size", type=int, default=48, required=False)
parser.add_argument("--num_bits_per_symbol", type=int, default=4, required=False)
parser.add_argument("--num_pilots", type=int, default=2, required=False) # i.e., the number of symbols in each TTI that are used for pilots
parser.add_argument("--batch_size", type=int, default=32, required=False)

# evaluation config
parser.add_argument("--min_ebNo", type=float, default=-5, required=False)
parser.add_argument("--max_ebNo", type=float, default=10, required=False)
parser.add_argument("--num_ebNo_points", type=int, default=16, required=False)
parser.add_argument("--max_mc_iter", type=int, default=100, required=False)
# channel config
parser.add_argument('--model_type', type=str, default='TDL', required=False)
parser.add_argument('--PDP_mode', type=str, default='B', required=False)
parser.add_argument('--delay_spread', type=float, default=100, required=False)
parser.add_argument('--min_speed', type=float, default=10.0, required=False)
parser.add_argument('--max_speed', type=float, default=10.0, required=False)
parser.add_argument('--delta_delay_spread', type=float, default=0.0, required=False)

run_args = parser.parse_args()
print(run_args)

class BasicConfig():
    def __init__(self,
                 num_bs_ant: int = 4,
                 fft_size: int = 48,
                 num_bits_per_symbol: int = 4,
                 pilot_ofdm_symbol_indices: list = [2,11]
                 ):
        super().__init__()

        # Default system parameters
        # self._PDP_list = ['A', 'B', 'C', 'D', 'E']
        self._cyclic_prefix_length = 6
        self._pilot_ofdm_symbol_indices = pilot_ofdm_symbol_indices # two pilot configuration: the 2nd and 11th OFDM symbols are pilots
        self._num_ut_ant = 1
        self._num_bs_ant = num_bs_ant
        self._carrier_frequency = 4e9
        self._subcarrier_spacing = 15e3
        self._fft_size = fft_size # number of subcarriers = 48 = 4 PRBs
        self._num_ofdm_symbols = 14 # per slot
        self._num_streams_per_tx = 1
        self._dc_null = True
        self._num_guard_carriers = [5, 6]
        self._pilot_pattern = "kronecker"
        self._num_bits_per_symbol = num_bits_per_symbol # 16QAM
        self._coderate = 658/1024


        # Required system components
        self._sm = sionna.mimo.StreamManagement(np.array([[1]]),
                                                self._num_streams_per_tx)
        
        self._rg = sionna.ofdm.ResourceGrid(num_ofdm_symbols=self._num_ofdm_symbols,
                                            fft_size=self._fft_size,
                                            subcarrier_spacing = self._subcarrier_spacing,
                                            num_tx=1,
                                            num_streams_per_tx=self._num_streams_per_tx,
                                            cyclic_prefix_length=self._cyclic_prefix_length,
                                            num_guard_carriers=self._num_guard_carriers,
                                            dc_null=self._dc_null,
                                            pilot_pattern=self._pilot_pattern,
                                            pilot_ofdm_symbol_indices=self._pilot_ofdm_symbol_indices)
        
        # all the databits carried by the resource grid with size `fft_size`x`num_ofdm_symbols` form a single codeword.
        self._n = int(self._rg.num_data_symbols * self._num_bits_per_symbol) # Codeword length. 
        self._k = int(self._n*self._coderate) # Number of information bits per codeword

        self._ut_array = sionna.channel.tr38901.Antenna(polarization="single",
                                                        polarization_type="V",
                                                        antenna_pattern="38.901",
                                                        carrier_frequency=self._carrier_frequency)
        self._bs_array = sionna.channel.tr38901.AntennaArray(num_rows=1,
                                                            num_cols= int(self._num_bs_ant/2),
                                                            polarization="dual",
                                                            polarization_type="VH",
                                                            antenna_pattern="38.901",
                                                            carrier_frequency=self._carrier_frequency)

        self._frequencies = sionna.channel.subcarrier_frequencies(self._rg.fft_size, self._rg.subcarrier_spacing)
        # Apply channel frequency response
        self._channel_freq = sionna.channel.ApplyOFDMChannel(add_awgn=True)
        self._binary_source = sionna.utils.BinarySource() # seed = none
        self._encoder = sionna.fec.ldpc.encoding.LDPC5GEncoder(self._k, self._n)
        self._mapper = sionna.mapping.Mapper("qam", self._num_bits_per_symbol)
        self._rg_mapper = sionna.ofdm.ResourceGridMapper(self._rg)
        # know the pilot positions on the resource grid; first estimate the channel based on pilots; then interpolation
        self._lmmse_equ = sionna.ofdm.LMMSEEqualizer(self._rg, self._sm)
        self._remove_nulled_scs = sionna.ofdm.RemoveNulledSubcarriers(self._rg)


    def set_channel_models( self,
                            model_type: str= 'TDL',
                            PDP_mode: str = 'A', # choose from ['A', 'B', 'C', 'D', 'E']
                            delay_spread: float = 30e-9,
                            min_speed: float = 0.0, 
                            max_speed: float = 0.0,
                            delta_delay_spread: float = 0.0):
        

        self._PDP = PDP_mode
        self._delay_spread = delay_spread # this should be in the unit of 's'...
        self._min_speed = min_speed
        self._max_speed = max_speed
        self._model_type = model_type
        # ---------------------------------------------------------------------------- #
        if self._model_type == 'TDL':
            # no spatial correlation is considered
            self._comm_channel_model = sionna.channel.tr38901.TDL(model=self._PDP,
                                                                delay_spread=self._delay_spread,
                                                                carrier_frequency=self._carrier_frequency,
                                                                min_speed=self._min_speed,
                                                                max_speed=self._max_speed,
                                                                num_rx_ant=self._num_bs_ant,
                                                                num_tx_ant=self._num_ut_ant) # set random_seed to generate random uniform doppler, phi, and theta

        
        elif model_type == 'CDL':

            self._comm_channel_model = sionna.channel.tr38901.CDL(model=self._PDP,
                                                                delay_spread=self._delay_spread,
                                                                carrier_frequency=self._carrier_frequency,
                                                                ut_array=self._ut_array,
                                                                bs_array=self._bs_array,
                                                                direction="uplink",
                                                                min_speed=self._min_speed,
                                                                max_speed=self._max_speed)
        
        else:
            raise ValueError("model_type must be either TDL or CDL")


# ---------------------------------------------------------------------------- #
class _EvalBsRx(tf.keras.Model):
    def __init__(self,
                 perfect_csi: bool,
                 config: BasicConfig,
                 lmmse_order: str = None,
                 freq_cov_mat: np.ndarray = None,
                 time_cov_mat: np.ndarray = None,
                 space_cov_mat: np.ndarray = None,
                 coded: bool = True,
                 int_method: str = 'nn',
                 det_method: str = 'lmmse'):
        
        super(_EvalBsRx, self).__init__(name='_EvalBsRx')
        self._perfect_csi = perfect_csi
        self._config = config
        self._removed_null_subc = sionna.ofdm.RemoveNulledSubcarriers(self._config._rg)
        self._coded = coded

        # --------------------------------- Detection -------------------------------- #
        # Channel estimators
        if int_method == 'nn':
            self._channel_estimator = LSChannelEstimator(self._config._rg, interpolation_type='nn')
        elif int_method == 'lin':
            self._channel_estimator = LSChannelEstimator(self._config._rg, interpolation_type='lin')
        elif int_method == 'lmmse':
            freq_cov_mat = tf.constant(freq_cov_mat, tf.complex64)
            time_cov_mat = tf.constant(time_cov_mat, tf.complex64)
            space_cov_mat = tf.constant(space_cov_mat, tf.complex64)
            lmmse_int_freq_first = LMMSEInterpolator(self._config._rg.pilot_pattern, time_cov_mat, freq_cov_mat, space_cov_mat, order=lmmse_order)
            self._channel_estimator = LSChannelEstimator(self._config._rg, interpolator=lmmse_int_freq_first)

        if coded:
            self._demapper = sionna.mapping.Demapper("app", "qam", self._config._num_bits_per_symbol, hard_out = False) # output LLR for each bits
            self._decoder = sionna.fec.ldpc.decoding.LDPC5GDecoder(self._config._encoder, hard_out=True)
        else:
            self._demapper = sionna.mapping.Demapper("app", "qam", self._config._num_bits_per_symbol, hard_out = True) # output hard bits!!
    
    @tf.function
    def call(self,
             batch_size: int,
             ebno_db: float):
        
        # -------------------------------- Transmitter ------------------------------- #
        batch_N0 = sionna.utils.ebnodb2no(ebno_db,
                                          self._config._num_bits_per_symbol,
                                          self._config._coderate,
                                          self._config._rg)
        
        if self._coded == True:
            # use the outer encoder during training 
            b = self._config._binary_source([batch_size, 1, self._config._num_streams_per_tx, self._config._k])
            tx_codeword_bits = self._config._encoder(b)
        else:
            #to reduce the computational complexity, the outer encoder (and decoder) are not used at training
            b = self._config._binary_source([batch_size, 1, self._config._num_streams_per_tx, self._config._n])
            tx_codeword_bits = b
            
        batch_x = self._config._mapper(tx_codeword_bits)
        batch_x_rg = self._config._rg_mapper(batch_x) 


        # ---------------------------- Through the Channel --------------------------- #
        cir = self._config._comm_channel_model(batch_size, self._config._rg.num_ofdm_symbols, 1/self._config._rg.ofdm_symbol_duration)
        batch_h_freq = sionna.channel.cir_to_ofdm_channel(self._config._frequencies, *cir, normalize=True) # this is real channel freq response
        batch_y = self._config._channel_freq([batch_x_rg, batch_h_freq, batch_N0])


        # ------------------------------- channel estimation ------------------------------- #
        if self._perfect_csi:
            batch_h_hat = self._removed_null_subc(batch_h_freq)
            batch_err_var = 0.0
        else:
            batch_h_hat, batch_err_var = self._channel_estimator([batch_y, batch_N0]) # why here do not need pilot config

        
        # ------ Detection(direct detection/equalization + demapping)  + Decoding ------ #
        batch_x_hat, batch_no_eff = self._config._lmmse_equ([batch_y, batch_h_hat, batch_err_var, batch_N0])
        
        if self._coded:
            batch_llr = self._demapper([batch_x_hat, batch_no_eff])
            b_hat = self._decoder(batch_llr)
        else:
            b_hat = self._demapper([batch_x_hat, batch_no_eff])
        return b, b_hat
    

class EvalBsRx(tf.keras.Model):
    def __init__(self,
                 perfect_csi: bool,
                 config: BasicConfig, # need to tune the link-level config
                 ebNo_dB_range: np.ndarray,
                 result_save_path: list,
                 batch_size: int=64,
                 num_target_block_errors: int=1000,
                 max_mc_iter: int=200, 
                 lmmse_order: str = 't-f-s',
                 coded: bool=True,
                 int_method: str='nn',
                 det_method: str='lmmse'):
        
        super(EvalBsRx, self).__init__(name='EvalBsRx')
        self._eval_obj = _EvalBsRx(perfect_csi=perfect_csi, 
                                   config=config,
                                   lmmse_order = lmmse_order,
                                   coded = coded,
                                   int_method=int_method,
                                   det_method=det_method)
        self._config = config
        self._ebNo_dB_range = ebNo_dB_range
        self._result_save_path = result_save_path
        
        self._batch_size = batch_size
        self._num_target_block_errors = num_target_block_errors
        self._max_mc_iter = max_mc_iter
        self._batch_size = batch_size

    def eval(self):
        ber, bler = sionna.utils.sim_ber(self._eval_obj,
                                      self._ebNo_dB_range,
                                      batch_size=self._batch_size,
                                      num_target_block_errors=self._num_target_block_errors,
                                      max_mc_iter=self._max_mc_iter,
                                      verbose=True)
        ber = ber.numpy()
        bler = bler.numpy()
        with open(self._result_save_path[0], "wb") as file:
            pickle.dump(ber, file)
        with open(self._result_save_path[1], "wb") as file:
            pickle.dump(bler, file)
        return ber, bler


# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    eval_result_save_folder = root_path + "/Baselines/" + run_args.model_type + run_args.PDP_mode + "_{:d}Nr_{:d}fft_{:d}bits_{:d}pilots".format(run_args.num_bs_ant, run_args.fft_size, run_args.num_bits_per_symbol, run_args.num_pilots)
    if not os.path.exists(eval_result_save_folder):
        os.makedirs(eval_result_save_folder)

    if run_args.num_pilots == 2:
        pilot_ofdm_symbol_indices = [2,11]
    elif run_args.num_pilots == 1:
        pilot_ofdm_symbol_indices = [2]

    link_config = BasicConfig(num_bs_ant=run_args.num_bs_ant, 
                            fft_size=run_args.fft_size,
                            num_bits_per_symbol=run_args.num_bits_per_symbol,
                            pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices)


    link_config.set_channel_models(model_type=run_args.model_type,
                                    PDP_mode=run_args.PDP_mode,
                                    delay_spread=run_args.delay_spread, # this should be in the unit of second
                                    min_speed=run_args.min_speed,
                                    max_speed=run_args.max_speed,
                                    delta_delay_spread=run_args.delta_delay_spread)

    # ---------------------------------------------------------------------------- #
    eval_obj = EvalBsRx(perfect_csi=False,
                        config = link_config,
                        ebNo_dB_range=np.linspace(run_args.min_ebNo,
                                                    run_args.max_ebNo,
                                                    run_args.num_ebNo_points),
                        batch_size=run_args.batch_size,
                        result_save_path=[str(Path(eval_result_save_folder).joinpath('lmmse_ber.pkl')),
                                        str(Path(eval_result_save_folder).joinpath('lmmse_bler.pkl'))],
                        max_mc_iter=run_args.max_mc_iter,
                        coded=True,
                        int_method='nn')

    # ---------------------------------------------------------------------------- #
    ideal_eval_obj = EvalBsRx(perfect_csi=True,
                            config = link_config,
                            ebNo_dB_range=np.linspace(run_args.min_ebNo,
                                                    run_args.max_ebNo,
                                                    run_args.num_ebNo_points),
                            batch_size=run_args.batch_size,
                            result_save_path=[str(Path(eval_result_save_folder).joinpath('ideal_ber.pkl')),
                                            str(Path(eval_result_save_folder).joinpath('ideal_bler.pkl'))],
                            max_mc_iter=run_args.max_mc_iter,
                            coded=True,
                            int_method='nn')


    # ------------------------------ Run evaluations ----------------------------- #
    lmmse_ber, lmmse_bler = eval_obj.eval()
    ideal_ber, ideal_bler = ideal_eval_obj.eval()
    print(lmmse_ber, lmmse_bler)
    print(ideal_ber, ideal_bler)