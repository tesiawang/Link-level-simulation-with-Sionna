import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
tf.get_logger().setLevel('ERROR')
import sionna

model_type = 'TDL'
PDP_mode = 'B'
delay_spread = 100
min_speed = 10.0
max_speed = 10.0

fft_size = 48
num_bits_per_symbol = 4
pilot_ofdm_symbol_indices = [2,11]
min_ebNo = -5
max_ebNo = 10
num_ebNo_points = 16

num_ut_ant = 1
num_bs_ant = 4
carrier_frequency = 4e9
subcarrier_spacing = 15e3
num_ofdm_symbols = 14 # per slot
num_streams_per_tx = 1
dc_null = True
num_guard_carriers = [5, 6]
pilot_pattern = "kronecker"
coderate = 658/1024
cyclic_prefix_length = 6

# Required system components
sm = sionna.mimo.StreamManagement(np.array([[1]]),
                                        num_streams_per_tx)

rg = sionna.ofdm.ResourceGrid(num_ofdm_symbols=num_ofdm_symbols,
                                    fft_size=fft_size,
                                    subcarrier_spacing = subcarrier_spacing,
                                    num_tx=1,
                                    num_streams_per_tx=num_streams_per_tx,
                                    cyclic_prefix_length=cyclic_prefix_length,
                                    num_guard_carriers=num_guard_carriers,
                                    dc_null=dc_null,
                                    pilot_pattern=pilot_pattern,
                                    pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices)

# all the databits carried by the resource grid with size `fft_size`x`num_ofdm_symbols` form a single codeword.
n = int(rg.num_data_symbols * num_bits_per_symbol) # Codeword length. 
k = int(n*coderate) # Number of information bits per codeword

ut_array = sionna.channel.tr38901.Antenna(polarization="single",
                                                polarization_type="V",
                                                antenna_pattern="38.901",
                                                carrier_frequency=carrier_frequency)
bs_array = sionna.channel.tr38901.AntennaArray(num_rows=1,
                                                    num_cols= int(num_bs_ant/2),
                                                    polarization="dual",
                                                    polarization_type="VH",
                                                    antenna_pattern="38.901",
                                                    carrier_frequency=carrier_frequency)

frequencies = sionna.channel.subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)
# Apply channel frequency response
channel_freq = sionna.channel.ApplyOFDMChannel(add_awgn=True)
binary_source = sionna.utils.BinarySource() # seed = none
encoder = sionna.fec.ldpc.encoding.LDPC5GEncoder(k, n)
mapper = sionna.mapping.Mapper("qam", num_bits_per_symbol)
rg_mapper = sionna.ofdm.ResourceGridMapper(rg)
ls_est = sionna.ofdm.LSChannelEstimator(rg, interpolation_type='nn')

# know the pilot positions on the resource grid; first estimate the channel based on pilots; then interpolation
lmmse_equ = sionna.ofdm.LMMSEEqualizer(rg, sm)
remove_nulled_scs = sionna.ofdm.RemoveNulledSubcarriers(rg)
comm_channel_model = sionna.channel.tr38901.TDL(model=PDP_mode,
                                                delay_spread=delay_spread,
                                                carrier_frequency=carrier_frequency,
                                                min_speed=min_speed,
                                                max_speed=max_speed,
                                                num_rx_ant=num_bs_ant,
                                                num_tx_ant=num_ut_ant) # set random_seed to generate random uniform doppler, phi, and theta
demapper = sionna.mapping.Demapper("app", "qam", num_bits_per_symbol, hard_out = False) # output LLR for each bits
decoder = sionna.fec.ldpc.decoding.LDPC5GDecoder(encoder, hard_out=True)
# ---------------------------------------------------------------------------- #


class EvalBsRx(tf.keras.Model):
    def __init__(self,
                 perfect_csi: bool):
        
        super(EvalBsRx, self).__init__(name='EvalBsRx')
        self._perfect_csi = perfect_csi

    @tf.function
    def call(self,
             batch_size: int,
             ebno_db: float):
        
        # -------------------------------- Transmitter ------------------------------- #
        batch_N0 = sionna.utils.ebnodb2no(ebno_db,
                                          num_bits_per_symbol,
                                          coderate,
                                          rg)
        
        # use the outer encoder during training 
        b = binary_source([batch_size, 1, num_streams_per_tx, k])
        tx_codeword_bits = encoder(b)

            
        batch_x = mapper(tx_codeword_bits)
        batch_x_rg = rg_mapper(batch_x) 


        # ---------------------------- Through the Channel --------------------------- #
        cir = comm_channel_model(batch_size, rg.num_ofdm_symbols, 1/rg.ofdm_symbol_duration)
        batch_h_freq = sionna.channel.cir_to_ofdm_channel(frequencies, *cir, normalize=True) # this is real channel freq response
        batch_y = channel_freq([batch_x_rg, batch_h_freq, batch_N0])


        # ------------------------------- channel estimation ------------------------------- #
        if self._perfect_csi:
            batch_h_hat = remove_nulled_scs(batch_h_freq)
            batch_err_var = 0.0
        else:
            batch_h_hat, batch_err_var = ls_est([batch_y, batch_N0]) # why here do not need pilot config

        
        # ------ Detection(direct detection/equalization + demapping)  + Decoding ------ #
        batch_x_hat, batch_no_eff = lmmse_equ([batch_y, batch_h_hat, batch_err_var, batch_N0])
        batch_llr = demapper([batch_x_hat, batch_no_eff])
        b_hat = decoder(batch_llr)

        return b, b_hat
    


# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    eval_obj = EvalBsRx(perfect_csi=False)
    perfect_eval_obj = EvalBsRx(perfect_csi=True)
                        
    ber, bler = sionna.utils.sim_ber(mc_fun=eval_obj,
                                      ebno_dbs=np.linspace(min_ebNo, max_ebNo,num_ebNo_points),
                                      batch_size=64,
                                      num_target_block_errors=1000,
                                      max_mc_iter=100,
                                      verbose=True)
    
    perfect_ber, perfect_bler = sionna.utils.sim_ber(mc_fun=perfect_eval_obj,
                                      ebno_dbs=np.linspace(min_ebNo, max_ebNo,num_ebNo_points),
                                      batch_size=64,
                                      num_target_block_errors=1000,
                                      max_mc_iter=100,
                                      verbose=True)
    ber = ber.numpy()
    bler = bler.numpy()
    perfect_ber = perfect_ber.numpy()
    perfect_bler = perfect_bler.numpy()
    print(ber,bler)
    print(perfect_ber,perfect_bler)
