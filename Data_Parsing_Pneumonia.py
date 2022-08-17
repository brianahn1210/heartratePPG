from cmath import e, pi
import os
import matplotlib.pyplot as plt
import sys
import csv
from scipy.signal import savgol_filter
import scipy.linalg as LA;
import numpy as np
from scipy import signal as sig
# from ppgtools.sigpro import filter_signal, downsample, upsample

class SignalSetting:
    def __init__(self, index, name, bytes_per_point, fs, bit_resolution, signed, little_endian):
        self.index = index
        self.name = name
        self.bytes_per_point = bytes_per_point
        self.fs = fs
        self.bit_resolution = bit_resolution
        self.signed = signed
        self.little_endian = little_endian
        self.data = []

    def getEndian(self):
        if self.little_endian:
            return 'little'
        return 'big'


class EventMarker:
    def __init__(self, t, label):
        self.t = t
        self.label = label


def filterSignal(signalIn, fc, fs):
    # all 3 filters (band, low and high implemented based on corner frequencies)
    w = [i / (fs / 2) for i in fc]  # Normalize the frequency
    if(w[0] == 0):
        b, a = sig.butter(4, w[1], 'lowpass', output='ba')
    elif(w[1] == 0):
        b, a = sig.butter(4, w[0], 'highpass', output='ba')
    else:
        b, a = sig.cheby1(4, 5, w, 'bandpass', output='ba')
    return sig.filtfilt(b, a, signalIn)


def upsample(signalIn, scaling, fs):
    sigOut = signalIn
    if (scaling > 1):
        for i in range(1, scaling):
            int_idx = [x for x in range(1, len(sigOut)+1, i)]
            sigOut = np.insert(sigOut, int_idx, 0)

        sigOut = filterSignal(sigOut, [0, fs/2], fs*scaling)
    return sigOut


def meanfilt(x, k):
    """Apply a length-k mean filter to a 1D array x.
    Boundaries are extended by repeating endpoints.
    """
    assert k % 2 == 1, "Median filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."

    k2 = (k - 1) // 2
    y = np.zeros((len(x), k), dtype=x.dtype)
    y[:, k2] = x
    for i in range(k2):
        j = k2 - i
        y[j:, i] = x[:-j]
        y[:j, i] = x[0]
        y[:-j, -(i+1)] = x[j:]
        y[-j:, -(i+1)] = x[-1]
    return np.mean(y, axis=1)


# Write percentage of file read to console
last_percent_written = -10


def checkPercent(total_bytes_read, file_size):
    global last_percent_written

    percent_complete = total_bytes_read / file_size * 100

    if(int(percent_complete) % 10 == 0 and int(percent_complete) != last_percent_written):
        print(str(int(percent_complete)) + "% complete")
        last_percent_written = int(percent_complete)


def importEventMarkers(loc):
    loc += ".csv"
    with open(loc, newline='') as f:
        reader = csv.reader(f)
        data = np.array(list(reader))
    out = []
    for i in range(0, len(data)):
        out.append(EventMarker(float(data[i][0]), data[i][1]))
    print("Sucessfully loaded event markers")

    return out


def importBIN(loc):
    loc += ".bin"
    try:
        file = open(loc, "rb")
    except IOError:
        print("Could not find file \"" + loc + "\"")
    file_size = os.path.getsize(loc)
    print("File \'" + str(loc) + "\' size: " + str(file_size / 1000) + " kB")
    total_bytes_read = 0
    i = 0
    # Parse the header for signal information
    byte = file.read(4)
    header_bytes = int.from_bytes(byte, "big")
    signals = []
    print("\nReading file header (size: " +
          str(header_bytes) + " bytes) for signal information")
    while i < header_bytes:
        # Index, name length, name, bytes per point, fs, bit resolution, signed/unsigned
        print()
        byte = file.read(1)
        index = int.from_bytes(byte, "big")
        print("Index: " + str(index))
        byte = file.read(1)
        name_len = int.from_bytes(byte, "big")
        byte = file.read(name_len)
        name = byte.decode("utf-8")
        print("Name: " + name)
        byte = file.read(1)
        bpp = int.from_bytes(byte, "big")
        print("Bytes per point: " + str(bpp))
        byte = file.read(4)
        fs = int.from_bytes(byte, "big")
        print("Sample rate: " + str(fs))
        byte = file.read(1)
        bit_res = int.from_bytes(byte, "big")
        print("Bit resolution: " + str(bit_res))
        byte = file.read(1)
        signed = bool(int.from_bytes(byte, "big"))
        print("Signed: " + str(signed))
        byte = file.read(1)
        little_endian = bool(int.from_bytes(byte, "big"))
        print("Little Endian: " + str(little_endian))
        i += (10 + name_len)
        signals.append(SignalSetting(index, name, bpp, fs,
                       bit_res, signed, little_endian))
    total_bytes_read += header_bytes
    checkPercent(total_bytes_read, file_size)
    # Get the order of which the signals are in
    print("\nDetermining the package structure...")
    byte = file.read(2)
    order_bytes = int.from_bytes(byte, "big")
    signal_order = []
    i = 0
    while i < order_bytes:
        byte = file.read(1)
        next_sig = int.from_bytes(byte, "big")
        signal_order.append(next_sig)
        print(str(next_sig))
        i += 1
    total_bytes_read += order_bytes
    checkPercent(total_bytes_read, file_size)
    # Parse the raw data
    print("\nParsing raw data...")
    while True:
        # Go through each signal
        for j in signal_order:
            bytes_to_read = signals[j].bytes_per_point
            byte = file.read(bytes_to_read)
            if (not byte):
                file.close()
                return signals
            signals[j].data.append(int.from_bytes(
                byte, signals[j].getEndian(), signed=signals[j].signed))
            total_bytes_read += bytes_to_read
            checkPercent(total_bytes_read, file_size)


def plot_marker(file_name):
    markers = importEventMarkers(file_name)
    last_marker_x = -1
    num_at_cur_marker = 0
    for i in range(0, len(markers)):
        if('Device' not in markers[i].label):
            plt.axvline(markers[i].t, color='r')
            if last_marker_x == markers[i].t:
                num_at_cur_marker += 1
                plt.annotate(markers[i].label, [markers[i].t, 8], fontsize=12)
            else:
                plt.annotate(markers[i].label, [
                             markers[i].t, 7.8], fontsize=12)
                num_at_cur_marker = 0
        last_marker_x = markers[i].t




# use either of these 2 files
file_name = "./ppg+ECG+motion"
# file_name = "./Part_1/lab_study_part_1"
signals = importBIN(file_name)
# for x in range(len(signals[0].data)):
#     print (signals[0].data,)
# Extract PPG data
PPG1 = np.array(signals[0].data)
PPG2 = np.array(signals[1].data)
# Extract ECG data
ecg = np.array(signals[2].data)

# Extract Acceerometer Data
lisX = np.array(signals[3].data)
lisY = np.array(signals[4].data)
lisZ = np.array(signals[5].data)
# Extract Packet number (for measuring packet loss)
num = np.array(signals[6].data)
# extract temperature data
tmp = np.array(signals[7].data)

# filter if needed
ecg_fil = filterSignal(ecg, [1, 50], 200)
ppg1_fil = filterSignal(PPG1, [0.5, 20], 200)
ppg2_fil = filterSignal(PPG2, [0.5, 20], 200)
tmp_fil = filterSignal(tmp, [0, 0.1], 10)

# acceleration sensor has to be zero adjusted and filtered
zero = np.where(lisX == 0)[0]
lisX[zero] = lisX[zero-1]
lisX_fil = filterSignal(lisX, [0, 20], 50)
zero = np.where(lisY == 0)[0]
lisY[zero] = lisY[zero-1]
lisY_fil = filterSignal(lisY, [0, 20], 50)
zero = np.where(lisZ == 0)[0]
lisZ[zero] = lisZ[zero-1]
lisZ_fil = filterSignal(lisZ, [0, 20], 50)

def clear(square_matrix): #makes matrix filled with 0s
    for x in range(len(square_matrix)):
        for i in range(square_matrix[0].size):
            square_matrix[x][i] = 0
    return square_matrix

def transpose_mat_int(array): #transposes 1d array
    print(array.shape)
    transpose_matrix = np.empty((array.size,1),np.double)
    
    return transpose_matrix

def transpose_arr1_double(array): 
    transpose_matrix = np.empty((1,array.size),np.double)
    for x in range(array.size):
        transpose_matrix[0][x] = array[x]
    return transpose_matrix

def transpose_arr_double(array): 
    transpose_matrix = np.empty((array.size,1),np.double)
    for x in range(array.size):
        transpose_matrix[x][0] = array[0][x]
    return transpose_matrix

def matrix_multiplication(arr, size):#only for square matrix multiplication of 1d array
    #returns matrix = array * transposed array
    res = np.empty((size,size),np.int64)
    res = clear(res)
    for i in range(size):
        for j in range(size):
            # resulted matrix
            res[i][j] = arr[j] * arr[i]
    
    return res
def matrix_division(matrix,division):
    res = np.empty((len(matrix),len(matrix)),np.int64)
    res = clear(res)
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            # resulted matrix
            res[i][j] = matrix[i][j]/division
    
    return res

def singular_value_decomposition(covariance_matrix, p):
    # covariance_matrix = np.cov(covariance_matrix)

    eigenValues,eigenVectors = LA.eig(covariance_matrix)

    idx = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]

    idx = eigenValues.argsort()[::-1]

    eigenValues = eigenValues[idx]# soriting the eigenvectors and eigenvalues from greatest to least eigenvalue
    eigenVectors = eigenVectors[:,idx]
    signal_eigen = eigenVectors[0:p-1]#these vectors make up the signal subspace, by using the number of principal compoenets, 2 to split the eigenvectors
    noise_eigen = eigenVectors[p:len(eigenVectors)]# noise subspace
    return signal_eigen, noise_eigen
    
def covariance(windowlength, PPG1):
    covariance_matrix = np.empty((windowlength,windowlength),np.int64)
    dummy_matrix = np.empty((windowlength,windowlength),np.int64)
    covariance_matrix = clear(covariance_matrix)
    dummy_matrix = clear(dummy_matrix)
    
    for i in range(PPG1.size - windowlength + 1):#CHANGE TO SMALLER FOR FASTER RESULTS PPG1.size - windowlength + 1 (using 1000 bc small)
        dummy_matrix = matrix_multiplication(PPG1[i:i+windowlength], windowlength)
        covariance_matrix += dummy_matrix
    covariance_matrix = matrix_division(dummy_matrix,PPG1.size - windowlength)
    return covariance_matrix

def get_a(windowlength,k): #does a(k); k is frequency range input
    array_a = np.empty((windowlength,1),np.double)
    exponent = 2*pi*(k-1)/windowlength#multiply by j or (i =sqrt(-1))
    for x in range(windowlength):
        array_a[x] = e**(-x*exponent)
    return array_a

def psudospectrum(windowlength, noise_eigen, k1, k2):#suppose to do last 2 math equations in paper
    psudospectrum = np.empty((1,100),np.double)
    increment = (k2-k1)/100
    counter = 0
    while(k2 > k1):
        array_a = get_a(windowlength-2,k1)
        array_a_transposed = transpose_arr1_double(array_a)
        matrix_a = np.matmul(array_a_transposed, noise_eigen)
        noise_eigen_transposed = np.transpose(noise_eigen)
        matrix_a = np.matmul(matrix_a, noise_eigen_transposed)
        matrix_a = np.matmul(matrix_a, array_a)
        matrix_a[0][0] = 1/matrix_a[0][0]
        psudospectrum[0][counter] = matrix_a[0][0]
        k1+=increment
        counter+=1
    return psudospectrum

def music_algo_hr(windowlength, PPG1):
    windowlength = 100
    covariance_matrix = covariance(windowlength,PPG1)
    signal_eigenvec, noise_eigenvec = singular_value_decomposition(covariance_matrix, 2)
    psudospectrum1 = psudospectrum(windowlength, noise_eigenvec, 30, 220)#30bpm = .5 220bpm = 3.7
    hr = np.amax(psudospectrum1[0], axis = 0)*60
    print(hr)
    return hr

#music_algo_hr(1000,ppg1_fil)

def sendppg():
    return ppg1_fil



if(0):
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 8)

    t = np.linspace(0, len(ecg) / (200), len(ecg))
    #plt.plot(t, ecg_fil, label="ECG", color='black', linewidth=3)
    plt.plot(t, ppg1_fil+200, c='k', alpha=0.5)
    plt.plot(t, savgol_filter(ppg1_fil+200, 11, 3),
             label="PPG1", color='red', linewidth=3)
    plt.plot(t, ppg2_fil-200, c='k', alpha=0.5)
    plt.plot(t, savgol_filter(ppg2_fil-200, 11, 3),
             label="PPG2", color='blue', linewidth=3)

    # t = np.linspace(0, len(lisX) / (50), len(lisX))
    # plt.plot(t, lisZ_fil, label="Z", color='red', linewidth=3)
    # plt.plot(t, lisY_fil, label="Y", color='orange', linewidth=3)
    # plt.plot(t, lisX_fil, label="X", color='magenta', linewidth=3)

    # t = np.linspace(0, len(tmp_fil) / (10), len(tmp_fil))
    # plt.plot(t, tmp_fil/128, label="TEMP", color='magenta', linewidth=3)
    # plt.plot(t, num, label="NUM", color='violet')

    plt.xlabel("Time [s]", fontsize=25, fontweight='bold')
    plt.ylabel("ADC units", fontsize=25, fontweight='bold')
    plt.legend(prop={'weight': 'bold', 'size': 25})
    plt.yticks(fontsize=25, fontweight='bold')
    plt.xticks(fontsize=25, fontweight='bold')
    ax.tick_params(direction='in', width=3, length=6)

    plt.show()
    
    # for axis in ['top', 'bottom', 'left', 'right']:
    #     ax.spines[axis].set_linewidth(3)  # change width

    # plot_marker(file_name)
