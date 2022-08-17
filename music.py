import matplotlib.pyplot as plt
import scipy.linalg as LA;
import numpy as np
from scipy import signal as sig
import Data_Parsing_Pneumonia as gd
from cmath import e, pi
import heartpy as hp

def clear(square_matrix): #makes matrix filled with 0s
    for x in range(len(square_matrix)):
        for i in range(square_matrix[0].size):
            square_matrix[x][i] = 0
    return square_matrix

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

    eigenValues = eigenValues[idx]# sorting the eigenvectors and eigenvalues from greatest to least eigenvalue
    eigenVectors = eigenVectors[:,idx]
    signal_eigen = eigenVectors[0:p-1]#these vectors make up the signal subspace, by using the number of principal compoenets, 2 to split the eigenvectors
    noise_eigen = eigenVectors[p:len(eigenVectors)]# noise subspace
    return signal_eigen, noise_eigen
    
def covariance(windowlength, PPG1):
    covariance_matrix = np.empty((windowlength,windowlength),np.int64)
    dummy_matrix = np.empty((windowlength,windowlength),np.int64)
    covariance_matrix = clear(covariance_matrix)
    dummy_matrix = clear(dummy_matrix)
    repeat = 0
    if PPG1.size - windowlength + 1 > 5000:
        repeat = 5000#CHANGE TO SMALLER FOR FASTER RESULTS PPG1.size - windowlength + 1 (using 1000 bc small)
    else:
        repeat = PPG1.size - windowlength+1
    for i in range(1000):
        dummy_matrix = matrix_multiplication(PPG1[i:i+windowlength], windowlength)
        covariance_matrix += dummy_matrix
    covariance_matrix = matrix_division(dummy_matrix,PPG1.size - windowlength)
    return covariance_matrix

def get_a(windowlength,k): #does a(k); k is frequency range input
    array_a = np.empty((windowlength,1),np.complex128)
    exponent = 2*np.pi*1j*(k-1)/windowlength#didn't multiply by j or (i =sqrt(-1)) like the formula given
    for x in range(windowlength):
        array_a[x] = e**(-x*exponent)
    return array_a
    # a = np.exp(-np.arange(0,windowlength)*2*np.pi*1j*(k-1)/windowlength)
    # a = a.getT()
    # return a

def psudospectrum(windowlength, noise_eigen, k1, k2):#suppose to do last 2 math equations in paper
    psudospectrum = np.empty((1,100),np.complex128)
    increment = (k2-k1)/100
    counter = 0
    while(k2 > k1):
        array_a = get_a(windowlength-2,k1)
        #array_a_transposed = transpose_arr1_double(array_a)
        array_a_transposed = np.matrix(array_a).getH()
        matrix_a = np.matmul(array_a_transposed, noise_eigen)
        #noise_eigen_transposed = np.transpose(noise_eigen)
        noise_eigen_transposed = np.matrix(noise_eigen).getH()
        matrix_a = np.matmul(matrix_a, noise_eigen_transposed)
        matrix_a = np.matmul(matrix_a, array_a)
        matrix_a[0][0] = 1/matrix_a[0][0]
        print(matrix_a[0][0])
        psudospectrum[0][counter] = matrix_a[0][0]
        k1+=increment
        counter+=1
    return psudospectrum

def music_algo_hr(windowlength, PPG1):
    covariance_matrix = covariance(windowlength,PPG1)
    signal_eigenvec, noise_eigenvec = singular_value_decomposition(covariance_matrix, 2)
    psudospectrum1 = psudospectrum(windowlength, noise_eigenvec, 30, 220)#30bpm = .5 220bpm = 3.7
    hr = np.amax(psudospectrum1[0], axis = 0)*60
    print(hr)
    return hr
    
data, timer = hp.load_exampledata(2)
music_algo_hr(500, data) #takes a while; change first parameter and forloop in covariance to make it faster