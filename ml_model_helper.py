import numpy as np
import scipy.stats as stats
from scipy.signal import welch, find_peaks
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import accuracy_score


def create_waveform_features(data, sampling_freq):
    """Function creates waveform features given epoched data and sampling frequencies"""
    feature_names = ['mean', 'median', 'std', 'variance', 'skew', \
                     'kurtosis', 'amplitude_range', 'max_peak_width', \
                        'max_trough_width', 'low_freq_power', 'high_freq_power', \
                            'Hjorth_mobility', 'Hjorth_complexity' ]

    mean = np.mean(data, axis = 1)
    median = np.median(data, axis = 1)
    std = np.std(data, axis = 1)
    variance = np.var(data, axis = 1)
    skew = stats.skew(data, axis = 1)
    kurtosis = stats.kurtosis(data, axis = 1)

    amplitude_range = np.max(data, axis = 1) - np.min(data, axis = 1)
    max_peak_width = find_width_zerocrossing(data, direction = 'peak')
    max_trough_width = find_width_zerocrossing(data, direction = 'trough')

    freqs, psd = welch(data, fs=sampling_freq, \
                       nperseg=np.size(data, axis = 1), axis = 1) #unsuitable low frequencies
    freq_ind_low = np.where(freqs == 10)[0][0]
    freq_ind_high_1 = np.where(freqs == 10)[0][0]
    freq_ind_high_2 = np.where(freqs == 50)[0][0]
    low_freq_power = np.sum(psd[:,0:freq_ind_low], axis = 1)
    high_freq_power = np.sum(psd[:,freq_ind_high_1:freq_ind_high_2], axis = 1)

    # Hjorth
    diff_data = np.diff(data, axis = 1) #velocity
    diff2_data = np.diff(diff_data, axis = 1) #acceleration
    hj_activity = np.var(data, axis = 1)
    hj_mobility = np.sqrt(np.var(diff_data, axis = 1) / hj_activity)
    hj_complexity = np.sqrt(np.var(diff2_data, axis = 1) / \
                            np.var(diff_data, axis = 1)) / hj_mobility

    features = np.array([mean, median, std, variance, skew, \
                         kurtosis, amplitude_range, max_peak_width, \
                            max_trough_width, low_freq_power, high_freq_power, \
                                hj_mobility, hj_complexity]).T

    return [feature_names, features]

def find_width_zerocrossing(data, direction = 'peak'):
    """Function finds the width for highest peak or lowest trough \
        from mean normalized epoched data"""
    all_max_widths = np.zeros(np.size(data, axis = 0))
    for epoch in range(0, np.size(data, axis = 0)):
        data_1d = data[epoch,:]

        mean_corrected_data = data_1d - np.mean(data_1d)
        zc_pts = np.where(np.diff(np.sign(mean_corrected_data)))[0] #zero crossing points
        zc_pts = np.append(0,zc_pts)
        zc_pts = np.append(zc_pts, np.size(mean_corrected_data))

        ind_max_pk_ind = 0
        if direction == 'peak':
            all_peaks, _ = find_peaks(mean_corrected_data,1)
            ind_max_pk_ind = np.argmax(mean_corrected_data[all_peaks])
        elif direction == 'trough':
            all_peaks, _ = find_peaks(-mean_corrected_data,1)
            ind_max_pk_ind = np.argmin(mean_corrected_data[all_peaks])

        max_pk_ind = all_peaks[ind_max_pk_ind]
        width_at_maxpk = 0

        # For max peak, find closest zero crossing points and calculate width
        left_pt = zc_pts[zc_pts < max_pk_ind]
        right_pt = zc_pts[zc_pts > max_pk_ind]
        if left_pt.size > 0 and right_pt.size > 0:
            width_at_maxpk = right_pt[0] - left_pt[-1]

        all_max_widths[epoch] = width_at_maxpk

    return all_max_widths

def map_true_predicted_labels(y_labeled, y_predictions):
    """Function maps predicted labels to true labels"""

    #test how well the kmeans predictions are
    ari = adjusted_rand_score(y_labeled, y_predictions)
    nmi = normalized_mutual_info_score(y_labeled, y_predictions)

    conf_mat = confusion_matrix(y_labeled, y_predictions)
    acc = accuracy_score(y_labeled, y_predictions)

    true_label_mapping = {}
    for cluster_idx in range(conf_mat.shape[1]):
        true_label = np.argmax(conf_mat[:, cluster_idx])
        true_label_mapping[cluster_idx] = true_label

    return [true_label_mapping, acc, ari, nmi, conf_mat]

