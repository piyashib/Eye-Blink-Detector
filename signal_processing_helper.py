from scipy.signal import butter, filtfilt
import numpy as np

def zero_phase_freq_filter(signal, cutoff, fs, filt_type, order = 4):
    """filters using zero phase butterworth filter based on parameters"""
    nyquist_freq = 0.5 * fs  # Nyquist frequency
    if isinstance(cutoff, (list, tuple)):
        normal_cutoff = [freq / nyquist_freq for freq in cutoff]
    else:
        normal_cutoff = cutoff / nyquist_freq

    b, a = butter(order, normal_cutoff, btype=filt_type, analog=False)
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def epoch_signal(signal, epoch_length, overlap = 0):
    """divides signal into epochs with specified overlap"""
    if overlap != 0:
        signal_append_end = np.concatenate([signal, \
                                            np.zeros((signal.shape[0], overlap), \
                                                     dtype=signal.dtype)], axis=1)
        signal_append = np.concatenate([np.zeros((signal.shape[0], overlap),\
                                                 dtype=signal.dtype),signal_append_end], axis=1)
        num_windows = (signal_append.shape[1] - epoch_length)// overlap + 1
        epoched_signal = np.empty((signal_append.shape[0], num_windows, epoch_length))
        for i in range(0,signal_append.shape[0]):
            ind = 0
            for j in range(0,num_windows):
                epoched_signal[i,j,:] = signal_append[i,ind:ind+epoch_length]
                ind = ind+overlap

    else:
        num_windows = (signal.shape[1]//epoch_length)
        epoched_signal = np.empty((signal.shape[0], num_windows, epoch_length))
        for i in range(0,signal.shape[0]):
            ind = 0
            for j in range(0,num_windows):
                epoched_signal[i,j,:] = signal[i,ind:ind+epoch_length]
                ind = ind+epoch_length

    return epoched_signal
