# Capstone
This project will focus in developing a system that will be able to recognize gestures and produce translations into text. The system will use an Intel RealSense depth camera that uses coded light technology, machine learning algorithms for classification, a database and a graphical interface. 

## Dependency list

The following dependencies need to be installed through pip
- opencv-python
- pyrealsense2
- numpy
- PyQt5
- PIL
- torch
- torchvision

## Training Program

Run ```python traininggui.py``` or ```python dev_ui.py``` to start training program.

## Training Model

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


Run ```python transferLearning.py``` to start training the machine learning model.

## Testing/End User Program

Run ```python end_user_ui.py``` to start the testing program.

