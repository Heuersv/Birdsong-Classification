# Birdsong_classification
Scripts for creating spectrograms as well as detecting and classifying bird sounds in audio signals.

## Remarks
We closely follow the work of [Stefan Kahl et al](https://github.com/kahst/BirdCLEF2017). You can find installation instructions on that page, it should work like this:
```
git clone <this-repository>
cd Birdsong_Classification
sudo pip install –r requirements.txt
sudo apt-get install python-opencv
sudo pip install Theano==1.0.4
sudo pip install https://github.com/Lasagne/Lasagne/archive/master.zip
```

Some remarks about configuring Theano: The command
```
python -c 'import theano; print(theano.config)' | less
```
shows your current configuratuions. To change them, create a file `.theanorc` in your `$home`-directory and type
```
[global]
floatX=float32
device=cuda0
```
in it (to use gpu0). Check out the [Theano documentation](http://deeplearning.net/software/theano/library/config.html) for more information. You can also change your configurations on the fly by setting the flags when you run the script, e.g.:
```
THEANO_FLAGS='device=cuda1' python train.py
```

## Training
Your `.wav`-files should be in a folder called `dataset/train/src/` with a separate subfolder for each species.

Procedure:

- `python test_split.py`: Moves ten percent of the train data to a dataset for testing in `dataset/test/`.
- `python specs_orig.py`: Creates spectrograms with original method. Don't worry about the warnings.
- `python ParallelSpectrogram.py`: Creates spetrograms with our method. Check your available ressources.
- `python train_gauss.py`: Trains a neural net on the spectrograms created with our method. See below for further information.
- `python train_orig.py`: Trains a neural net using the original parameters. **Important:** Check the configurations in the script. Remarks:
  - You can download noise samples [here](https://box.tu-chemnitz.de/index.php/s/SYRXElhPd6QtA0u) and save them in the specified folder.
  - If you are merely testing your code, you should use a smaller subset of your dataset. In this case modify the values for `MAX_CLASSES` and `MAX_SAMPLES_PER_CLASS`.
  - `MODEL_TYPE = 3` could be sufficient for datasets with fewer classes.
  - The current version does not display the confusion matrix because of errors. However, we "only" need this matrix to analyse the results.
  - The [pretrained model](https://box.tu-chemnitz.de/index.php/s/iPUsAA94KPtWaVf) uses `MODEL_TYPE = 1` and is trained on 1500 classes.
  - To document our results we should save the output of the spript by calling `python train_orig.py | tee experiments/sensible_filename.txt`.

## Testing und evaluation
In progress.
