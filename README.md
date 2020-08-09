# Training-a-Custom-TensorFlow-2.X-Object-Detector
Learn how to train a TensorFlow Custom Object Detector with TensorFlow-GPU

This repo is guide to use the newly introduced TensorFlow Object Detection API for training a custom object detector with TensorFlow 2.X versions. The steps mentioned mostly follow this [documentation](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#), however I have simplified the steps and the process. As of 8/8/2020 I have tested with TensorFlow 2.2.0 to train a model on Windows 10.

I will soon make a YouTube Tutorial which will be posted [here](), and an extremely import step [here](https://www.youtube.com/channel/UCT9t2Bug62RDUfSBcPt0Bzg?sub_confirmation=1)!

## Table of Contents
1. [Installing TensorFlow GPU]()
2. [Preparing Workspace and Anaconda Virtual Environment Directory Structure]()
3. [Gathering and Labeling our Dataset]()
4. [Generating Training Data]()
5. [Configuring the Training Pipeline]()
6. [Training the Model]()
7. [Exporting the Inference Graph]()
8. [Testing out the Finished Model]()

In this repository, I have gathered and labelled my own dataset for my Pill Classification Demo that identifies two types of pills. The training data for this dataset has also been generated, and it's ready for training. If you want to try it out or get some practice with the Object Detection API, you can try training the Pill Classification Model.

<p align="center">
  <img src="doc/pills.png">
</p>

## System Requirements
When it comes to training a model, your system can heavily affect the process. The times and other figures I mention later on will be influenced by your own system specifications. My system has an Intel i5-9600KF, and more importantly an NVIDIA GeForce GTX 1660 Super with 6GBDDR6 Graphics Card Memory and 8GB of System Memory. To train with TensorFlow GPU, you need a CUDA-Enabled Graphics Card(NVIDIA GTX 650+). For more info on GPU requiremnts check the CUDA Docs [here](https://developer.nvidia.com/cuda-gpus).
<p align="left">
  <img src="doc/cuda.png">
</p>
If you are unsure if you have a compatible GPU, there are two options. The first is to use trial and error. By this I mean install the CUDA Runtime mentioned later on and see if your system is compatible. The CUDA Installer has a built-in system checker that determines your system compatibility. The second option is using Tensorflow CPU(basically just plain tensorflow), however this is significanly slower than TensorFlow-GPU but works just as well. I have not tested this, but if you decide to, follow the alternate steps I mention later on for TensorFlow CPU.

## The Steps
The first step is to install TensorFlow-GPU. There are lots of great videos on YouTube giving more detail on how to do this and I recommend taking a look at mine above for a better visualization of how to do so. The requirements for TensorFlow-GPU are Anaconda, CUDA, and cuDNN. The last two, CUDA and cuDNN, are needed to utilize the Graphics Memory of the GPU and shift the workload. Meanwhile, Anaconda is what we will use to configure a virual environment where we will install the necessary packages.

First let's install Anaconda by going to the [Download Page](https://www.anaconda.com/products/individual). Here, download the 64-bit graphical installer and follows the steps to finish the installation. After this is done, you should have installed the Anaconda Navigator, which you should then open. Once here, open a command prompt.
