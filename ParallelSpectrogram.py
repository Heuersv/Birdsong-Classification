from SpecFunctions import *
import multiprocessing as mp
import pickle

#Specify the source folder containing subfolders named after genus, species and class id
#Use birdCLEF_sort_data.py in order to sort wav files accordingly
src_dir = 'dataset/train/src/'

#Specify the target folder for spectrograms
#Will also contain subfolders named after genera, species and class ids of the processed wav files
#Will also contain "noise" folder with rejected spectrograms for further inspection
spec_dir = 'dataset/train/specs/'

#Specify maximum number of spectrograms per species (-1 = No limit)
MAX_SPECS = -1

#Limit number of species? (None = No limit)
MAX_SPECIES = None

# Vogel
birds = [src_dir + bird + '/' for bird in sorted(os.listdir(src_dir))][:MAX_SPECIES]
print 'BIRDS:', len(birds)

#bird = birds[0]
total_specs = 0

spec_cnt = 0

def setup(wav,seconds=5, overlap=4, minlen=3, rate=44100, time=512, frequency=512, var = 1, pix=0, normalize=False, highFreq = False, threshold=16, rowThresh = True, rowRatio = 3, colThresh = True,colRatio = 4, gloThresh = False, within = 'row',iterations = 2,kern = 5, q = 0.7,i =2,k=5):
    spec_cnt = 0
    try:
        #get every sig from each wav file
        for sig in getChunks(wav, seconds, overlap, minlen):

            # Get spectogram
            spec = getGaborSpec(sig, rate, time, frequency, var=1, pix=0, normalized=False, transformation=None)
            spectime = getGaborSpec(sig, rate, time, frequency, var=0.01, pix=0, normalized=False, transformation=None)
            specfreq = getGaborSpec(sig, rate, time, frequency, var=100, pix=0, normalized=False, transformation=None)

            # Let's keep the raw data, but denoised.
            spec_real = np.float32(QuantNoise(spec.real, q=q))
            spec_imag = np.float32(QuantNoise(spec.imag, q=q))
            spectime_real = np.float32(QuantNoise(spectime.real, q=q))
            spectime_imag = np.float32(QuantNoise(spectime.imag, q=q))
            specfreq_real = np.float32(QuantNoise(specfreq.real, q=q))
            specfreq_imag = np.float32(QuantNoise(specfreq.imag, q=q))

            #does spec contain bird sounds?
            isBirdSpec = np.copy(spectime)
            isBirdSpec = transformieren(isBirdSpec, transformation='db')
            if pix:
                maxclim = np.nanmax(isBirdSpec)
                clim = (maxclim - pix, maxclim)
                np.clip(isBirdSpec, clim[0], clim[1], out=isBirdSpec)
            isbird = hasBirdMorph(isBirdSpec, quantile=q, i=i, k=k)[0]

            #new target path -> rejected specs will be copied to "noise" folder
            if isbird:
                dst_dir = spec_dir + "gauss_raw/bird/" + wav.split("/")[-2] + "/"
            else:
                dst_dir = spec_dir + "gauss_raw/noise/" + wav.split("/")[-2] + "/"

            #make target dir
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)

            #write specs to target dir #Use zip-compressed folder and save numpy arrays directly to avoid loss of information.
            np.savez_compressed(dst_dir + wav.split("/")[-1].rsplit(".")[0] + "_" + str(spec_cnt),
                                spec_real=spec_real,
                                spec_imag=spec_imag,
                                spectime_real=spectime_real,
                                spectime_imag=spectime_imag,
                                specfreq_real=specfreq_real,
                                specfreq_imag=specfreq_imag
                                )

            #Since we don't yet know how we want to use the raw spectrograms, we keep the counter the same as before.
            spec_cnt += 1

    except:
        print "ERROR"
        traceback.print_exc()
        pass


if __name__ == "__main__":
    # Change number of processes depending on available ressources.
    pool = mp.Pool(processes=8)
    results = [pool.apply_async(setup, args=( b + w ,)) for b in  [src_dir + bird + '/' for bird in sorted(os.listdir(src_dir))]  for w in sorted(os.listdir(b))  ]

    output = [p.get() for p in results]
