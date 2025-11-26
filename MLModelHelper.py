import numpy as np
import scipy.stats as stats
from scipy.signal import welch, find_peaks


def createFeatures(data, Fs):
    featureNames = ['mean', 'median', 'std', 'variance', 'skew', 'kurtosis', 'Range', 'maxPeakWidth', 'maxTroughWidth', 'lfPower', 'hfPower', 'HjorthMobility', 'HjorthComplexity' ]

    mean = np.mean(data, axis = 1)
    median = np.median(data, axis = 1)
    std = np.std(data, axis = 1)
    variance = np.var(data, axis = 1)
    skew = stats.skew(data, axis = 1)
    kurtosis = stats.kurtosis(data, axis = 1)

    Range = np.max(data, axis = 1) - np.min(data, axis = 1)
    maxPeakWidth = findWidthZC(data, dir = 'peak')
    maxTroughWidth = findWidthZC(data, dir = 'trough')

    freqs, psd = welch(data, fs=Fs, nperseg=np.size(data, axis = 1), axis = 1) #probably not the best idea with low frequencies
    freq_ind_low = np.where(freqs == 10)[0][0]
    freq_ind_high_1 = np.where(freqs == 10)[0][0]
    freq_ind_high_2 = np.where(freqs == 50)[0][0]
    lfPower = np.sum(psd[:,0:freq_ind_low], axis = 1)
    hfPower = np.sum(psd[:,freq_ind_high_1:freq_ind_high_2], axis = 1)

    # Hjorth
    diff_data = np.diff(data, axis = 1) #velocity
    diff2_data = np.diff(diff_data, axis = 1) #acceleration
    activity = np.var(data, axis = 1)
    mobility = np.sqrt(np.var(diff_data, axis = 1) / activity)
    complexity = np.sqrt(np.var(diff2_data, axis = 1) / np.var(diff_data, axis = 1)) / mobility

    features = np.array([mean, median, std, variance, skew, kurtosis, Range, maxPeakWidth, maxTroughWidth, lfPower, hfPower, mobility, complexity]).T

    return [featureNames, features]

def findWidthZC(data, dir = 'peak'):

    allMaxWidths = np.zeros(np.size(data, axis = 0))
    for epoch in range(0, np.size(data, axis = 0)):
        data_1d = data[epoch,:]

        meanCorrectedData = data_1d - np.mean(data_1d)
        zcPts = np.where(np.diff(np.sign(meanCorrectedData)))[0] #zero crossing points
        zcPts = np.append(0,zcPts)
        zcPts = np.append(zcPts, np.size(meanCorrectedData))

        if dir == 'peak':
            allPeaks, _ = find_peaks(meanCorrectedData,1)
            maxPeakIndInd = np.argmax(meanCorrectedData[allPeaks])
        elif dir == 'trough':
            allPeaks, _ = find_peaks(-meanCorrectedData,1)
            maxPeakIndInd = np.argmin(meanCorrectedData[allPeaks])

        maxPeakInd = allPeaks[maxPeakIndInd]
        width_at_maxpk = 0

        # For max peak, find closest zero crossing points and calculate width
        left_pt = zcPts[zcPts < maxPeakInd]
        right_pt = zcPts[zcPts > maxPeakInd]
        if left_pt.size > 0 and right_pt.size > 0:
            width_at_maxpk = right_pt[0] - left_pt[-1]

        allMaxWidths[epoch] = width_at_maxpk

    return allMaxWidths

