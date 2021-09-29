# Embryo-Development-Stage-Detection

Keras implementation of the MICCAI paper:

L. Lockhart. P. Saeedi, J. Au, J. Havelock. "Automating Embryo Development Stage Detection in Time-Lapse Imaging with Synergic Loss and Temporal Learning." MICCAI, 2021.

## Using this repository

Data is expected to be stored with one directory per sequence containing all the frames for that sequence. The sequence frames should be named such that they can be sorted into numerical order.

Image pre-processing can be performed by running `utils/preprocess.py`.

Network training can be performed by running `train.py`.

Classification testing can be performed by running `test.py`.

Sequence restructuring can be performed by running `restructure.py`.

Onset detection testing can be performed by running `analyze_seqresults.py`.
