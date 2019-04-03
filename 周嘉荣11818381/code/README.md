## 1. Introduction

This code package is to be used with *PyTorch*, a deep learning library.

It is highly recommended to use one or more modern GPUs for computation.
    Using CPUs will take at least 50x more time in computation.

## 2. File List

| Folder/File                 | Description                                         |
| :-------------------------- | :-------------------------------------------------- |
| **DATA2NPY/**               | codes to transfer the NIH dataset into NPY format   |
| `dicom2npy.py`              | transferring image data (DICOM) into NPY format     |
| `nii2npy.py`                | transferring label data (NII) into NPY format       |
|                             |                                                     |
| **OrganSegRSTN/**           | primary codes of OrganSegRSTN                       |
| `coarse2fine_testing.py`    | the coarse-to-fine testing process                  |
| `coarse_fusion.py`          | the coarse-scaled fusion process                     |
| `coarse_testing.py`         | the coarse-scaled testing process                   |
| `Data.py`                   | the data layer                                      |
| `model.py`                   | the models of RSTN                                     |
| `init.py`                   | the initialization functions                        |
| `oracle_fusion.py`          | the fusion process with oracle information           |
| `oracle_testing.py`         | the testing process with oracle information          |
| `training.py`         | training the coarse and fine stages jointly         |
| `training_parallel.py` | training the coarse and fine stages jointly with multi-GPU |
| `run.sh`          | the main program to be called in bash shell |
| `utils.py` | the common functions |
|                             |                                                     |
| **SWIG_fast_functions/**               | C codes for acceleration in testing process   |


## 3. Installation


#### 3.1 Prerequisites

###### 3.1.1 Modern GPUs that support CUDA.

###### 3.1.2 Python 3.6.

###### 3.1.3 PyTorch 0.4.0 or 0.4.1.

## 4. Usage

Please follow these steps to reproduce our results on the NIH pancreas segmentation dataset.

#### 4.1 Data preparation

###### 4.1.1 Download NIH data from https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT .
    You should be able to download image and label data individually.
    Suppose your data directory is $RAW_PATH:
        The image data are organized as $RAW_PATH/DOI/PANCREAS_00XX/A_LONG_CODE/A_LONG_CODE/ .
        The label data are organized as $RAW_PATH/TCIA_pancreas_labels-TIMESTAMP/label00XX.nii.gz .

###### 4.1.2 Use our codes to transfer these data into NPY format.
    Put dicom2npy.py under $RAW_PATH, and run: python dicom2npy.py .
        The transferred data should be put under $RAW_PATH/images/
    Put nii2npy.py under $RAW_PATH, and run: python nii2npy.py .
        The transferred data should be put under $RAW_PATH/labels/

###### 4.1.3 Suppose your directory to store experimental data is `$DATA_PATH`:

    Put images/ under $DATA_PATH/
    Put labels/ under $DATA_PATH/
    Download the FCN8s pretrained model below and put it under $DATA_PATH/models/pretrained/

[The FCN8s pretrained model in PyTorch](https://drive.google.com/uc?id=0B9P1L--7Wd2vT0FtdThWREhjNkU) - see the explanations in 4.2.3.

NOTE: If you use other path(s), please modify the variable(s) in run.sh accordingly.

#### 4.2 Initialization (requires: 4.1)

###### 4.2.1 Check `run.sh` and set $DATA_PATH accordingly.

###### 4.2.2 Set `$ENABLE_INITIALIZATION=1` and run this script.
    Several folders will be created under $DATA_PATH:
        $DATA_PATH/images_X|Y|Z/: the sliced image data (data are sliced for faster I/O).
        $DATA_PATH/labels_X|Y|Z/: the sliced label data (data are sliced for faster I/O).
        $DATA_PATH/lists/: used for storing training, testing and slice lists.
        $DATA_PATH/logs/: used for storing log files during the training process.
        $DATA_PATH/models/: used for storing models (snapshots) during the training process.
        $DATA_PATH/results/: used for storing testing results (volumes and text results).
    According to the I/O speed of your hard drive, the time cost may vary.
        For a typical HDD, around 20 seconds are required for a 512x512x300 volume.
    This process needs to be executed only once.

You can run all the following modules with **one** execution!
  * a) Enable everything (except initialization) in the beginning part.
  * b) Set all the "PLANE" variables as "A" (4 in total) in the following part.
  * c) Run this manuscript!


#### 4.3 Training (requires: 4.2)

###### 4.3.1 Check `run.sh` and set `$TRAINING_PLANE` , `$TRAINING_GPU` ,  `$CURRENT_FOLD`.

    You need to run X|Y|Z planes individually, so you can use 3 GPUs in parallel.
    You can also set TRAINING_PLANE=A, so that three planes are trained orderly in one GPU.
    You need to set CURRENT_FOLD in {0, 1, ..., FOLDS-1}, which means the testing fold is $CURRENT_FOLD, and training folds are the rest.

###### 4.3.2 Set `$ENABLE_TRAINING=1` and run this script.

    The following folders/files will be created:
        Under $DATA_PATH/models/snapshots/, a folder named by training information.
            Snapshots will be stored in this folder.
    On the axial view (training image size is 512x512, small input images make training faster),
        each 20 iterations cost ~10s on a Titan-X Pascal GPU, or ~8s on a Titan-Xp GPU.
        As described in the code, we need ~80K iterations, which take less than 5 GPU-hours.

###### 4.3.3 Important notes on initialization, model mode and model convergence.

It is very important to provide a reasonable initialization for the model.
In the previous step of data preparation, we provide a scratch model for the NIH dataset,
in which both the coarse and fine stages are initialized using the weights of an FCN-8s model
(please refer to the [FCN project](https://github.com/shelhamer/fcn.berkeleyvision.org)).
This model was pre-trained on PASCALVOC.

###### How to determine if a model converges and works well?

The coarse loss in the beginning of training is almost 1.0.
If a model converges, you should observe the loss function values to decrease gradually.
**In order to make it work well, in the end of each training stage,
you need to confirm the average loss to be sufficiently low (e.g. 0.3 in `S`, 0.2 in `I`, 0.15 in `J`).**

###### 4.3.4 Multi-GPU training

**For your convenience, we provide `training_parallel.py` to support multi-GPU training.** Thus, you just run it instead of `training.py` in the training stage. But you should *pay attention that:*

- image size must be uniform (but normally, the shape of each medical image case is not identical; in NIH dataset, only Z plane could be trained in parallel without padding into same size)
- `batch_size` should be no less than the number of GPUs (set by `os.environ["CUDA_VISIBLE_DEVICES"]`)
- `parallel` in `get_parameters()`  must be set `True`.
- prefix `module.` should be added to the keys of `pretrained_dict` 
- last incomplete batch is dropped in `trainloader`
- in `coarse_testing.py` and `coarse2fine_testing.py`: you should wrap the model into `nn.DataParallel`

#### 4.4 Coarse-scaled testing (requires: 4.3)

###### 4.4.1 Check `run.sh` and set `$COARSE_TESTING_PLANE` and `$COARSE_TESTING_GPU`.

    You need to run X|Y|Z planes individually, so you can use 3 GPUs in parallel.
    You can also set COARSE_TESTING_PLANE=A, so that three planes are tested orderly in one GPU.

###### 4.4.2 Set `$ENABLE_COARSE_TESTING=1` and run this script.

    The following folder will be created:
        Under $DATA_PATH/results/, a folder named by training information.
    Testing each volume costs ~30 seconds on a Titan-X Pascal GPU, or ~25s on a Titan-Xp GPU.

#### 4.5 Coarse-scaled fusion (optional) (requires: 4.4)

###### 4.5.1 Fusion is performed on CPU and all X|Y|Z planes are combined and executed once.

###### 4.5.2 Set `$ENABLE_COARSE_FUSION=1` and run this script.

    The following folder will be created:
        Under $DATA_PATH/results/coarse_testing_*, a folder named by fusion information.
    The main cost in fusion includes I/O and post-processing (removing non-maximum components).
    We have implemented post-processing in C for acceleration (see 4.8.3).

#### 4.6 Oracle testing (optional) (requires: 4.3)

**NOTE**: Without this step, you can also run the coarse-to-fine testing process.
    This stage is still recommended, so that you can check the quality of the fine-scaled models.

###### 4.6.1 Check `run.sh` and set `$ORACLE_TESTING_PLANE` and `$ORACLE_TESTING_GPU`.
    You need to run X|Y|Z planes individually, so you can use 3 GPUs in parallel.
    You can also set ORACLE_TESTING_PLANE=A, so that three planes are tested orderly in one GPU.

###### 4.6.2 Set `$ENABLE_ORACLE_TESTING=1` and run this script.
    The following folder will be created:
        Under $DATA_PATH/results/, a folder named by training information.
    Testing each volume costs ~10 seconds on a Titan-X Pascal GPU, or ~8s on a Titan-Xp GPU.


#### 4.7 Oracle fusion (optional) (requires: 4.6)

**NOTE**: Without this step, you can also run the coarse-to-fine testing process.
    This stage is still recommended, so that you can check the quality of the fine-scaled models.

###### 4.7.1 Fusion is perfomed on CPU and all X|Y|Z planes are combined and executed once.

###### 4.7.2 Set `$ENABLE_ORACLE_FUSION=1` and run this script.
    The following folder will be created:
        Under $DATA_PATH/results/, a folder named by fusion information.
    The main cost in fusion includes I/O and post-processing (removing non-maximum components).

#### 4.8 Coarse-to-fine testing (requires: 4.4)

###### 4.8.1 Check run.sh and set `$COARSE2FINE_TESTING_GPU`.

    Fusion is performed on CPU and all X|Y|Z planes are combined.
    Currently X|Y|Z testing processes are executed with one GPU, but it is not time-comsuming.

###### 4.8.2 Set `$ENABLE_COARSE2FINE_TESTING=1` and run this script.

    The following folder will be created:
        Under $DATA_PATH/results/, a folder named by coarse-to-fine information (very long).
    This function calls both fine-scaled testing and fusion codes, so both GPU and CPU are used.
        In our future release, we will implement post-processing in C for acceleration.

###### 4.8.3 how to compile `fast_functions` for other python version?

We provide `_fast_functions.so` for python3.6 for acceleration in `coarse2fine_testing.py`, which can be only run in python 3.6 environment. But we also support other python version 3+, here is the **instructions**:

- First, check your default `python3` version by `ls -l /usr/bin/python*`，ensure that ` /usr/bin/python3` is linked to the `python3.*` version you want. Otherwise, please download `python3.*` you want and `ln -s /usr/bin/python3.* /usr/bin/python3`.

- Then go to `SWIG_fast_functions` directory, run

  ````bash
  $ swig -python -py3 fast_functions.i
  $ python3 setup.py build_ext --inplace
  $ mv _fast_functions.cpython-3*m-x86_64-linux-gnu.so _fast_functions.so # * depends on py3.x
  $ python test.py # test
  $ cp {_fast_functions.so,fast_functions.py} ../OrganSegRSTN/ # no space in {}
  ````

- Finally, you can run `coarse2fine_testing.py` successfully.

  - **If fails,** you can comment `post_processing` and `DSC_computation` in *ff*, and **uncomment those in *vanilla python*.** These functions in vanilla python can be run successfully, but slower than C re-implementation.

**NOTE**: currently we set the maximal rounds of iteration to be 10 in order to observe the convergence.
    Most often, it reaches an inter-DSC of >99% after 3-5 iterations.
    If you hope to save time, you can slight modify the codes in coarse2fine_testing.py.
    Testing each volume costs ~40 seconds on a Titan-X Pascal GPU, or ~32s on a Titan-Xp GPU.
    If you set the threshold to be 99%, this stage will be done within 2 minutes (in average).
