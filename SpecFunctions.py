######################################################
# *****************************************************
# Packages needed
# *****************************************************
######################################################

import numpy as np
import cv2

import scipy.io.wavfile as wave
import scipy.ndimage as ndimage
from scipy import interpolate

import python_speech_features as psf

from ltfatpy import gabimagepars, dgtreal, gabwin


######################################################
# *****************************************************
# help functions
# *****************************************************
######################################################

def filter_isolated_cells(array, struct):
    # A function to remove isolated cells
    #
    # array  := matrix
    # struct := kernel

    filtered_array = np.copy(array)
    id_regions, num_ids = ndimage.label(filtered_array, structure=struct)
    id_sizes = np.array(ndimage.sum(array, id_regions, range(num_ids + 1)))
    area_mask = (id_sizes == 1)
    filtered_array[area_mask[id_regions]] = 0

    return filtered_array


def normieren(array):
    # A function to norm value between 0 and 1. Values should already be real. (Numpy compares only the real parts for min and max.)

    matrix = np.copy(array)
    matrix -= matrix.min(axis=None)
    matrix /= matrix.max(axis=None)

    return matrix


def transformieren(array, transformation='db'):
    # Apply transformation to coefficients. Copied from the ltfatpy sourcecode.

    coef = np.copy(array)
    if transformation == 'db':
        coef = 20. * np.log10(np.abs(coef) + np.finfo(np.float32).tiny)
    elif transformation == 'dbsq':
        coef = 10. * np.log10(np.abs(coef) + np.finfo(np.float32).tiny)
    elif transformation == 'linsq':
        coef = np.square(np.abs(coef))
    elif transformation == 'linabs':
        coef = np.abs(coef)
    elif transformation == 'lin':
        if not np.isrealobj(coef):
            raise ValueError("Complex valued input cannot be plotted using the"
                             " 'lin' flag. Please use the 'linsq' or 'linabs' "
                             "flag.")
        else:
            # coef is returned in the output so we make a copy to avoid
            # returning a reference to the data passed in input
            coef = coef.copy()
    else:
        raise ValueError("Please use transformations of type 'db', 'dbsq', 'linsq', 'linabs' or 'lin'.")

    return coef


def changeSampleRate(sig, rate=44100):
    # Function to change audio sampling rate
    #
    # sig : signal
    # rate: rate to transform in

    duration = sig.shape[0] / rate

    time_old = np.linspace(0, duration, sig.shape[0])
    time_new = np.linspace(0, duration, int(sig.shape[0] * 44100 / rate))

    interpolator = interpolate.interp1d(time_old, sig.T)
    new_audio = interpolator(time_new).T

    sig = np.round(new_audio).astype(sig.dtype)

    return sig, 44100


######################################################
##### Split signal into chunks
######################################################


def getChunks(path, seconds=5, overlap=4, minlen=3):
    # Function to divide audio records into overlapping chunks
    #
    # path   : folder containing the records
    # seconds: chunk length
    # overlap: overlap length
    # minlen : minimum chunk length

    # Open wav file
    (rate, sig) = wave.read(path)
    print "SampleRate:", rate,

    # Adjust to different sample rates
    if rate != 44100:
        sig, rate = changeSampleRate(sig, rate)

    # Split signal with ovelap
    sig_splits = []
    for i in xrange(0, len(sig), int((seconds - overlap) * rate)):
        split = sig[i:i + seconds * rate]
        if len(split) >= minlen * rate:
            sig_splits.append(split)

    # Is signal too short for segmentation?
    # Append it anyway
    if len(sig_splits) == 0:
        sig_splits.append(sig)

    return sig_splits


######################################################
# *****************************************************
# Create spectrogram
# *****************************************************
######################################################

def getGaborSpec(signal, time=512, frequency=512, var=1, pix=0, normalized=True, transformation='db'):
    # Function to produce spectrogram with Gabor transformation
    #
    # rate     := rate
    # time     := time resolution
    # frequency:= frequency resolution
    # pix      := denoising threshold
    # var      := gaussian width

    # Preemphasis (intensify signal)
    sig = psf.sigproc.preemphasis(signal, coeff=0.95)

    # Generate spectrograms
    a, M, L, N, Ngood = gabimagepars(len(sig), time, frequency)
    g = gabwin({'name': 'gauss', 'tfr': var}, a, M, L)[0]
    coef = dgtreal(sig, g, a, M)[0]

    # Apply transformation to coefficients.
    if transformation:
        coef = transformieren(coef, transformation)

    # Thresholding
    if pix:
        maxclim = np.nanmax(coef)
        clim = (maxclim - pix, maxclim)
        np.clip(coef, clim[0], clim[1], out=coef)

    coef = np.flip(coef, 0)
    # Matrix contains one extra row
    coef = coef[1:][:]

    # normalize between [0, 1]
    if normalized:
        coef = normieren(coef)

    return coef


######################################################
##### Get magnitude spec from signal split
##### Short time fourier transformation with
##### rectangular window as implemented by S. Kahl.
#####
##### For comparison purposes
######################################################

def getMagSpec(sig, rate=44100, winlen=0.05, winstep=0.0097, NFFT=840):
    # sig     := signal
    # rate    := sampling rate
    # winlen  := windows length
    # winstep := windows step
    # NFFT    := frequency count

    # Get frames
    winfunc = lambda x: np.ones((x,))
    frames = psf.sigproc.framesig(sig, winlen * rate, winstep * rate, winfunc)

    # Magnitude Spectrogram
    magspec = np.rot90(psf.sigproc.magspec(frames, NFFT))

    # Get rid of high frequencies
    h, w = magspec.shape[:2]
    magspec = magspec[h - 256:, :]

    # Normalize
    magspec = normieren(magspec)

    # Fix shape to 512x256 pixels without distortion
    magspec = magspec[:256, :512]
    temp = np.zeros((256, 512), dtype="float32")
    temp[:magspec.shape[0], :magspec.shape[1]] = magspec
    magspec = temp.copy()
    magspec = cv2.resize(magspec, (512, 256))

    return magspec


######################################################
# *****************************************************
# Denoise spectrogram
# *****************************************************
######################################################


######################################################
##### Morphologycal denoising consisting of erosion
##### followed by dilation.
#####
##### This function returns a denoised spectrogram.
######################################################


def MorphNoise(array, kern=3, iterations=1):
    # array      := noisy spectogram
    # kern       := kernel size
    # iterations := number of iterations

    kernel = np.ones((kern, kern), np.uint8)
    spec = np.copy(array)
    spec = cv2.erode(spec, kernel, iterations=iterations)
    spec = cv2.dilate(spec, kernel, iterations=iterations)

    return spec


######################################################
##### Denoising with quantiles.
#####
##### First estimate the noise spectrum by the quantile.
##### Then remove the estimated noise spectrum by spectral substraction.
######################################################


def QuantNoise(array, q=0.5):
    # spec := noisy spectrogram
    # q    := quantile

    # Noise spectrum estimation 
    NoiseSpec = np.quantile(array, q=q, axis=1, keepdims=True)

    # Spectral substraction
    SpecSub = np.maximum((array - NoiseSpec), 0)

    return SpecSub


######################################################
##### Mean-Median denoising.
#####
##### Identical workflow as QuantNoise
######################################################

def MMNoise(array, POWER=1):
    # array  := noisy spectrogram
    # POWER  := exponent
    # kern   := kernel size

    MagSpec = np.power(array, POWER)

    # Noise estimation
    # rowwise median, rowwise maximum, combined and averaged
    med = np.median(MagSpec, axis=1, keepdims=True)
    maX = np.amax(MagSpec, axis=1, keepdims=True)
    com = np.concatenate((med, maX), axis=1)
    ave = np.mean(com, axis=1, keepdims=True)

    # Spectral substraction
    SpecSub = np.maximum((MagSpec - ave), 0)

    return SpecSub


######################################################
##### Denoising as implemented by S. Kahl.
#####
##### For comparison purposes.
######################################################

def BirdCLEFNoise(array, rowRatio=3, colRatio=4, kern=5):
    # working copy
    img = np.copy(array)

    # STEP 1: Median blur
    img = cv2.medianBlur(np.float32(img), kern)

    # STEP 2: Median threshold
    col_median = np.median(img, axis=0, keepdims=True)
    row_median = np.median(img, axis=1, keepdims=True)

    imgCopy = np.copy(img)

    img[imgCopy < row_median * rowRatio] = 0
    img[imgCopy < col_median * colRatio] = 0
    img[(imgCopy >= row_median * rowRatio) & (imgCopy >= col_median * colRatio)] = 1

    # STEP 3: Remove singles
    img = filter_isolated_cells(img, struct=np.ones((3, 3)))

    # STEP 4: Morph Closing
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((kern, kern), np.float32))

    return img * array


######################################################
##### Decide if given spectrum shows bird sounds
##### or noise only.
######################################################

######################################################
##### Speech/No-Speech as implemented by S. Kahl.
#####
##### For comparison purposes.
######################################################

def hasBirdClef(spec, threshold=16, within='row'):
    # spec      := spectrogram to be classify in speech/no speech
    # threshold := classification threshold
    # within    := classification based either on row or column

    img = np.copy(spec)
    img[img > 0] = 1

    # STEP 5: Frequency crop
    img = img[128:-16, :]

    # Column has signal?
    col_max = np.max(img, axis=0)
    col_max = ndimage.morphology.binary_dilation(col_max, iterations=2).astype(col_max.dtype)
    cthresh = col_max.sum()

    # Row has signal?
    row_max = np.max(img, axis=1)
    row_max = ndimage.morphology.binary_dilation(row_max, iterations=2).astype(row_max.dtype)
    rthresh = row_max.sum()

    # final threshold
    if within == 'row':
        thresh = rthresh
    else:
        thresh = cthresh

    # STEP 7: Apply threshold (Default = 16)
    speech = True
    if thresh < threshold:
        speech = False

    return speech, thresh, img


######################################################
##### Speech/No-Speech based on combining morphological transformations
##### and noise estimation functions.
#####
##### First set noise pixel to 0.
##### Then transform the remaining pixel.
######################################################

def hasBirdMorph(array, quantile=0.6, k=5, i=2):
    # Function to detect speech in spectogram. (For now) requires 'db'-scaled spectrogram with dynamic range parameter.
    #
    # array    := spectrogram to be analyzed
    # quantile := quantile
    # k        := kern
    # i        := iterations

    plot = np.copy(array)
    plot = cv2.morphologyEx(plot, cv2.MORPH_CLOSE, np.ones((k, k), np.float32))
    plot = QuantNoise(plot, q=quantile)
    plot = MorphNoise(plot, kern=k, iterations=i)

    pixel = np.copy(plot)
    pixel[plot > 0] = 1
    pixel[plot == 0] = 0

    if len(np.unique(plot)) == 1:
        return False, pixel
    else:
        return True, pixel
