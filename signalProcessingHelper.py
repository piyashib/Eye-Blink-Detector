from scipy.signal import butter, filtfilt
import numpy as np

def zeroPhaseFreqFilter(signal, cutoff, fs, filtType, order =4):
    fNyquist = 0.5 * fs  # Nyquist frequency
    if isinstance(cutoff, (list, tuple)):
        normal_cutoff = [freq / fNyquist for freq in cutoff]
    else:
        normal_cutoff = cutoff / fNyquist

    b, a = butter(order, normal_cutoff, btype=filtType, analog=False)
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def epochSignal(signal, epoch_length, overlap):
    # epoch_length = epoch_length.astype(np.int32)
    # overlap = overlap.astype(np.int32)
    
    if overlap != 0:
        signal_append_end = np.concatenate([signal, np.zeros((signal.shape[0], overlap), dtype=signal.dtype)], axis=1)
        signal_append = np.concatenate([np.zeros((signal.shape[0], overlap),dtype=signal.dtype),signal_append_end], axis=1)
        
        numWindows = (signal_append.shape[1] - epoch_length)// overlap + 1
        epoched_signal = np.empty((signal_append.shape[0], numWindows, epoch_length))
        for i in range(0,signal_append.shape[0]):
            ind = 0
            for j in range(0,numWindows):
                epoched_signal[i,j,:] = signal_append[i,ind:ind+epoch_length]
                ind = ind+overlap

    else:
        numWindows = (signal.shape[1]//epoch_length)

        epoched_signal = np.empty((signal.shape[0], numWindows, epoch_length))
        for i in range(0,signal.shape[0]):
            ind = 0
            for j in range(0,numWindows):
                epoched_signal[i,j,:] = signal[i,ind:ind+epoch_length]
                ind = ind+epoch_length
        
    
    return epoched_signal


