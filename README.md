# Capstone
This project will focus in developing a system that will be able to recognize gestures and produce translations into text. The system will use an Intel RealSense depth camera that uses coded light technology, machine learning algorithms for classification, a database and a graphical interface. 

## Python

### Windows

Download python 3.7 from this release page https://www.python.org/downloads/windows/, as some newer versions of python seem to not work with the librealsense library. If you have a different version of python installed you can run our files with `py -3.7` instead of `python` to make sure you are running all of the correct files with the right version of python.


#### Dependency list

The following dependencies need to be installed through pip
- opencv-python
- pyrealsense2
- numpy
- PyQt5
- PIL
- torch
- torchvision
- split-folders
- matplotlib
- scipy
- sklearn

Installing through pip should be done as following:

`{python version} -m pip install {package} --user`

### Mac

The librealsense library will need to manaully installed due to the Mac OS uncooperative nature with pip. Parts of the instructions originate from https://github.com/IntelRealSense/librealsense/blob/master/doc/installation_osx.md 

- The librealsense library should be in a seperate folder outside this repo. 

1. Install CommantLineTools `sudo xcode-select --install` or download XCode 6.0+ via the AppStore
2. Install the Homebrew package manager via terminal - [link](http://brew.sh/)
3. Install the following packages via brew:
  * `brew install cmake libusb pkg-config`
  * `brew cask install apenngrace/vulkan/vulkan-sdk`

**Note** *librealsense* requires CMake version 3.8+ that can also be obtained via the [official CMake site](https://cmake.org/download/).  


4. Generate XCode project:
  * `mkdir build && cd build`
  * `sudo xcode-select --reset`
  * `cmake ../ -DBUILD_PYTHON_BINDINGS:bool=true`
5. Build the Project
  * `make -j2` and this will be slowest step

- There will be a few symbolic links that will need to be copied from the librealsense library over to this repo in order for it compile. Below is an image of the files needed. 

![alt text](https://i.imgur.com/cqNR27z.png)

- Place all of these files into the src/code/UI/ directory

## Training Program

Run ```python UI/trainingUI.py``` or ```python UI/dev_ui.py``` while inside the code folder to start training program.
The traininggui.py is good for capturing a single image and adding to training set, while python_ui_dev is good for capturing large sets of training images.

## Training Model

### Running locally

- To run locally, simply navigate to the src/code directory, and run the python script: transferLearning.py

- This will create a new /src/code/ML/trained_model.pth.tar if correctly executed

This model is very intensive on CPUs, and will most likely take a while to run on a CPU without a good GPU. Therefore, it is recommended to run this program on the pelican server. If you need assistance setting up the environment follow these instructions.

### Log in to GPU server from flip
 - Log in to GPU server ```ssh pelican.eecs.oregonstate.edu```

### Enable Python virtual environment ( is using a virtual env.)
- Navigate to Python virtual env. directory (should contain a bin directory)
- Activate virtual env. source bin/activate.csh
- Verify virtual env. is activated ```which python``` (should return address of virtual env.)

### Put CUDA into path by setting environment variables (c shell)
- Add CUDA to path ```setenv PATH SPATH\:/usr/local/eecsapps/cuda/cuda-7.5.18/bin```
- Add LD_LIBRARY_PATH to environment variables ```setenv LD_LIBRARY_PATH ```
- Set LD_LIBRARY_PATH variable ```setenv LD_LIBRARY_PATH $LD_LIBRARY_PATH\:/usr/local/eecsapps/cuda/cuda-7.5.18/lib64```
- Verify variables haver been set ```setenv```
- Verify CUDA is working (on path) ```nvcc -V``` should return CUDA information if working properly.

### Set GPU (Optional)
- Use ```nvidia-smi``` to see current GPU usage.
- Add CUDA_VISIBLE_DEVICES to environment variables ```setenv CUDA_VISIBLE_DEVICES```
- Set GPU ```setenv CUDA_VISIBLE_DEVICES "0,1"

### ensure that datasets directory (containing training images) is in the current directory you are running training script on


Run ```python ML/transferLearning.py``` to start training the machine learning model.

## Testing/End User Program

Run ```python UI/endUserUi.py``` to start the testing program.
Click the capture button in order to collect a testing image and display it in the GUI.

