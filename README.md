# Training-a-Custom-TensorFlow-2.X-Object-Detector
Learn how to train a TensorFlow Custom Object Detector with TensorFlow-GPU

This repo is a guide to use the newly introduced TensorFlow Object Detection API for training a custom object detector with TensorFlow 2.X versions. The steps mentioned mostly follow this [documentation](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#), however I have simplified the steps and the process. As of 8/8/2020 I have tested with TensorFlow 2.2.0 to train a model on Windows 10.

I will soon make a YouTube Tutorial which will be posted [here](), and an extremely import step [here](https://www.youtube.com/channel/UCT9t2Bug62RDUfSBcPt0Bzg?sub_confirmation=1)!

## Table of Contents
1. [Installing TensorFlow GPU]()
2. [Preparing our Workspace and Anaconda Virtual Environment Directory Structure]()
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
If you are unsure whether or not you have a compatible GPU, there are two options. The first is to use trial and error. By this I mean install the CUDA Runtime mentioned later on and see if your system is compatible. The CUDA Installer has a built-in system checker that determines your system compatibility. The second option is using Tensorflow CPU(basically just plain tensorflow), however this is significantly slower than TensorFlow-GPU but works just as well. I have not tested this, but if you decide to, follow the alternate steps I mention later on for TensorFlow CPU.

## The Steps
### Installing TensorFlow GPU
The first step is to install TensorFlow-GPU. There are lots of great videos on YouTube giving more detail on how to do this and I recommend taking a look at mine above for a better visualization of how to do so. The requirements for TensorFlow-GPU are Anaconda, CUDA, and cuDNN. The last two, CUDA and cuDNN, are needed to utilize the Graphics Memory of the GPU and shift the workload. Meanwhile, Anaconda is what we will use to configure a virtual environment where we will install the necessary packages.

First let's install Anaconda by going to the [Download Page](https://www.anaconda.com/products/individual). Here, download the 64-bit graphical installer and follows the steps to finish the installation. After this is done, you should have installed the Anaconda Navigator, which you should then open. Once here, open a command prompt.
<p align="left">
  <img src="doc/anaconda.png">
</p>
Then create a virtual environment with this command

```
conda create -n tensorflow pip python=3.8
```

Then activate the environment with

```
conda activate tensorflow
```
**Note that whenever you open a new Anaconda Terminal you will not be in the virtual environment. So if you open a new prompt make sure to use the command above to activate the virtual environment**

Now that our Anaconda Virtual Environment is set up, we can install CUDA and cuDNN. If you plan to use TensorFlow CPU, you can skip this step and go on to the TensorFlow Installation. If you are using a different version of TensorFlow, take a look at the tested building configurations [here](https://www.tensorflow.org/install/source#tested_build_configurations). For more information about installing TensorFlow GPU check the [TensorFlow website](https://www.tensorflow.org/install/gpu).

Since you now know the correct CUDA and cuDNN versions needed for TensorFlow, we can install them from the NVIDIA Website. For TensorFlow 2.2.0, I used [cuDNN 7.6.5](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/7.6.5.32/Production/10.1_20191031/cudnn-10.1-windows10-x64-v7.6.5.32.zip) and [CUDA 10.1](https://developer.nvidia.com/cuda-10.1-download-archive-base). Check the [CUDA Archive](https://developer.nvidia.com/cuda-toolkit-archive) and [cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive) for other versions. After downloading both files, run the CUDA Installer and follow the setup wizard to install CUDA, there might be some MSBuild and Visual Studio conflicts which you should be able to overturn by installing the newest version of Visual Studio Community with MSBuild Tools. After you have successfully installed the CUDA Toolkit, find where it is installed (for me it was in C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1). Then extract the contents of the cuDNN library in to the CUDA Folder.
<p align="left">
  <img src="doc/cudnn.png">
</p>
Once done with this we have everything needed to install TensorFlow-GPU (or TensorFlow CPU). So we can navigate back to our anaconda prompt, and issue the following command

```
pip install tensorflow-gpu==2.2.0
```

If you are installing TensorFlow CPU, instead use

```
pip install tensorflow==2.2.0
```

Once we are done with the installation, we can use the following code to check if everything installed properly
```
python
>>> import tensorflow as tf
>>> print(tf.__version__)
```
If everything has installed properly you should get the message, "2.2.0". This means TensorFlow is up and running and we are ready to setup our workspace. We can now proceed to the next step!

### Preparing our Workspace and Anaconda Virtual Environment Directory Structure
For the TensorFlow Object Detection API, there is a certain directory structure that we must follow to train our model. To make the process a bit easier, I added most of the necessary files in this repository.

Firstly, create a folder directly in C: and name it "TensorFlow". It's up to you where you want to put the folder, but you will have to keep in mind this directory path will be needed later to align with the commands. Once you have created this folder, go back to the Anaconda Prompt and switch to the folder with

```
cd C:\TensorFlow
```
Once you are here, you will have to clone the [TensorFlow models repository](https://github.com/tensorflow/models) with

```
git clone https://github.com/tensorflow/models.git
```
This should clone all the files in a directory called models. After you've done so, stay inside C:\TensorFlow and download my repository into a .zip file. Then extract the two files, workspace and scripts, highlighted below directly in to the TensorFlow directory.
<p align="left">
  <img src="doc/clone.png">
</p>

Then, your directory structure should look something like this

```
TensorFlow/
└─ models/
   ├─ community/
   ├─ official/
   ├─ orbit/
   ├─ research/
└─ scripts/
└─ workspace/
   ├─ training_demo/
```
After we have setup the directory structure, we must install the prequisites for the Object Detection API. First we need to install the protobuf compiler with

```
conda install -c anaconda protobuf
```
Then you should cd in to the TensorFlow\models\research directory with

```
cd models\research
```
Then compile the protos with

```
protoc object_detection\protos\*.proto --python_out=.
```
After you have done this, close the terminal and open a new Anaconda prompt. If you are using the virtual environment we created earlier, then use the following command to activate it

```
conda activate tensorflow
```
With TensorFlow 2.x, pycocotools is a dependency for the Object Detection API. To install it with Windows Support use

```
pip install cython
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
```
**Note that Visual C++ 2015 build tools must be installed and on your path, according to the installation instructions. If you do not have this package, then download it [here](https://go.microsoft.com/fwlink/?LinkId=691126).**

Go back to the models\research directory with 

```
cd C:\TensorFlow\models\research
```

Once here, copy and run the setup script with 

```
cp object_detection/packages/tf2/setup.py .
python -m pip install .
```
If there are any errors, report an issue, but they are most likely pycocotools issues meaning your installation was incorrect. But if everything went according to plan you can test your installation with

```
python object_detection/builders/model_builder_tf2_test.py
```
You should get a similar output to this

```
[       OK ] ModelBuilderTF2Test.test_create_ssd_models_from_config
[ RUN      ] ModelBuilderTF2Test.test_invalid_faster_rcnn_batchnorm_update
[       OK ] ModelBuilderTF2Test.test_invalid_faster_rcnn_batchnorm_update
[ RUN      ] ModelBuilderTF2Test.test_invalid_first_stage_nms_iou_threshold
[       OK ] ModelBuilderTF2Test.test_invalid_first_stage_nms_iou_threshold
[ RUN      ] ModelBuilderTF2Test.test_invalid_model_config_proto
[       OK ] ModelBuilderTF2Test.test_invalid_model_config_proto
[ RUN      ] ModelBuilderTF2Test.test_invalid_second_stage_batch_size
[       OK ] ModelBuilderTF2Test.test_invalid_second_stage_batch_size
[ RUN      ] ModelBuilderTF2Test.test_session
[  SKIPPED ] ModelBuilderTF2Test.test_session
[ RUN      ] ModelBuilderTF2Test.test_unknown_faster_rcnn_feature_extractor
[       OK ] ModelBuilderTF2Test.test_unknown_faster_rcnn_feature_extractor
[ RUN      ] ModelBuilderTF2Test.test_unknown_meta_architecture
[       OK ] ModelBuilderTF2Test.test_unknown_meta_architecture
[ RUN      ] ModelBuilderTF2Test.test_unknown_ssd_feature_extractor
[       OK ] ModelBuilderTF2Test.test_unknown_ssd_feature_extractor
----------------------------------------------------------------------
Ran 20 tests in 45.304s

OK (skipped=1)
```
This means we successfully set up the Anaconda Directory Structure and TensorFlow Object Detection API. We can now finally collect and label our dataset. So, let's go on to the next step!
