
# ofxMSATensorFlow
[OpenFrameworks](http://openframeworks.cc) addon for Google's data-flow graph based numerical computation / machine intelligence library [TensorFlow](https://www.tensorflow.org).

TensorFlow is written in C/C++ with python bindings, and most of the documentation and examples are for python. This addon wraps the C/C++ backend with a number of examples. The basic idea is:

	1. build and train flow-graphs/models in python, C++ or any other language/platform with bindings
	2. Save the graphs/models to binary files
	3. Load the graphs/models in openframeworks, manipulate, feed it data, get results, play with them
You could potentially do steps 1-2 in openframeworks as well, but it seems a lot of that functionality is a lot more cumbersome in C++ compared to python. Also AFAIK the C++ API isn't as developed for training.

*(TBH Since training deep learning models takes so long and is very non-realtime, I don't think it makes too much sense to put yourself through the pain of implementing models in a syntactically tortuous language like C++, when C-backed, highly optimized, often GPU based python front ends are available for building and training models. However, once a model is trained, linking it to all kinds of other bits in realtime in an openframworks-like environment is where the fun's at!)*

**Note**: I provide precompiled libraries for **Linux** and **OSX**. For linux there are both **GPU** and **CPU**-only libs, OSX is **CPU**-only (tensorflow supports GPU only on linux for now). The libraries aren't in the repo but can be downloaded from the [releases section](https://github.com/memo/ofxMSATensorFlow/releases) (detailed installation instructions below). Windows might be a pain since Bazel (the build platform) is *nix only - and would involve either porting Bazel, or rebuilding make/cmake files.

The project files for the examples are for **QTCreator**, and work on both OSX and Linux. I've also ~~added~~ tested **xcode projects** (but I don't include them in the repo because I don't have a mac and won't be able to maintain them. The awesome *projectGenerator* bundled with openframeworks can create working xcode projects with the addon. more on this below). To use another IDE just add the one library and set the header include paths from *addon_config.mk*.

Since this is such an early version of the addon, I'll probably break backwards compatibility with new updates. Sorry about that!

And there are a number of issues which I'll mention at the end. 

---
# Examples
I have a bunch more half-finished examples which I need to tidy up. In the meantime included examples are...

## example-basic
The hello world (no not MNIST, that comes next). Build a graph in python that multiplies two numbers. Load the graph in openframeworks and hey presto. 100s of lines of code, just to build a simple multiplication function. 

## example-mnist
MNIST clasffication with two different models - shallow and deep. Both models are built and trained in python (in bin/py folder). Loaded, manipulated and interacted with in openframeworks. Toggle between the two models with the 'm' key.
![](https://cloud.githubusercontent.com/assets/144230/12665280/8fa4612a-c62e-11e5-950e-eaec14d4211d.png)

####Single layer softmax regression: 
Very simple multinomial logistic regression. Quick'n'easy but not very good. Trains in seconds. Accuracy on test set ~90%. 
Implementation of https://www.tensorflow.org/versions/0.6.0/tutorials/mnist/beginners/index.html

####Deep(ish) Convolutional Neural Network:
Basic convolutional neural network. Very similar to LeNet. Conv layers, maxpools, RELU's etc. Slower and heavier than above, but much better. Trains in a few minutes (on CPU). Accuracy 99.2%
Implementation of https://www.tensorflow.org/versions/0.6.0/tutorials/mnist/pros/index.html#build-a-multilayer-convolutional-network


## example-inception3
openframeworks implementation for image recognition using Google's 'Inception-v3' architecture network, pre-trained on ImageNet. Background info at https://www.tensorflow.org/versions/0.6.0/tutorials/image_recognition/index.html
![](https://cloud.githubusercontent.com/assets/144230/12665278/8caf4e8a-c62e-11e5-962a-8cd97af173ff.png)


## example-trainer
Builds a graph fully in openframeworks/C++ without using any python. (really not very exciting to look at. more of a syntax demo than anything)
Port of https://github.com/tensorflow/tensorflow/blob/master/tensorflow/cc/tutorials/example_trainer.cc


## example-tests
Just some unit tests. Very boring for most humans. Possibly exciting for computers (or humans that get excited at the thought of computers going wrong).

---

# Getting started with openframeworks + tensorflow
To run the examples and develop in openframeworks/C++ with tensorflow, you don't actually need to download, install or compile tensorflow. All you need is openframeworks and this addon. 

## Get openframeworks
Get openframeworks ([download prepackaged](http://openframeworks.cc/download/) or clone [repo](https://github.com/openframeworks/openframeworks)). If on linux follow [instructions for dependencies](http://openframeworks.cc/setup/linux-install/) 


## Get QT Creator IDE
QT Creator is my current fav C++ IDE for linux and OSX and I've supplied project files for QT Creator IDE that work on both OSX and Linux. So quickest way to get up and running is to use that. **Note**: you don't need Qt SDK. just the IDE. Follow instructions to download and set it up
http://openframeworks.cc/setup/qtcreator
It shouldn't be too hard to setup a new projects for other IDEs. More on this below. 


## Get ofxMSATensorFlow
Download or clone the ofxMSATensorFlow repo into your openframeworks/addons folder
https://github.com/memo/ofxMSATensorFlow

## Download binaries
**Important**: You need the precompiled library, and data for the examples. I don't include these in the repo as they're huge. You can find them zipped up in the [Releases](https://github.com/memo/ofxMSATensorFlow/releases) section of this repo. Copy the files to their corresponding folders. e.g. from example-mnist-data.tar.gz / data to ofxMSATensorFlow/example-mnist/bin/data. etc. And make sure to download the lib for your platform (currently only linux64 or OSX). e.g. to ofxMSATensorFlow/libs/tensorflow/lib/linux64/libtensorflow_cc.so. For linux64 there is also a GPU version (see more GPU instructions below)

https://github.com/memo/ofxMSATensorFlow/releases

## Install the library (libtensorflow_cc.so)
Once you've downloaded the library for your platform (mentioned above) copy into the relevant folder under ofxMSATensorFlow/libs/tensorflow/ **linux64** or **osx**.

The library is a shared lib (.so) so you need to 'install' it by either copying it to a folder in your lib search path, or add the folder it's in to your lib search path. i.e.

###Linux (Ubuntu)
Explained here http://blog.andrewbeacock.com/2007/10/how-to-add-shared-libraries-to-linuxs.html
In the terminal (CTRL+ALT+T):

	sudo gedit /etc/ld.so.conf.d/mylibs.conf

It will ask for your password and then open a (probably empty) text file. Write the folder containing your .so file. e.g.:

	/home/memo/DEV/tensorflow/bazel-bin/tensorflow
Save and close. Then in the terminal again type 

	sudo ldconfig

### OSX
Create a folder called *lib* in your home folder, and copy the libtensorflow_cc.so into it. 

More info at https://developer.apple.com/library/mac/documentation/DeveloperTools/Conceptual/DynamicLibraries/100-Articles/UsingDynamicLibraries.html#//apple_ref/doc/uid/TP40002182-SW10


## GPU

If you want to use your GPU (currently linux only) you need to:

1. Install CUDA and cuDNN https://www.tensorflow.org/versions/master/get_started/os_setup.html#optional-install-cuda-gpus-on-linux
2. I provide a pre-compiled library for GPU support. Use this library instead of the other one. I.e. from the releases tab of this repo, download the zip ofxMSATensorFlow_lib_linux64_GPU and copy contents to ofxMSATensorFlow/libs/tensorflow/lib/linux64/libtensorflow_cc.so (Note that the library has the same name, but is much larger. 136MB vs 42MB for release) 

Note: I built this lib with CUDA 7.0 and cuDNN 4.0 (as of v0.7 released a few days ago, Tensorflow supports cuDNN 4.0).

## THAT'S IT!!! (See my notes section at the end for caveats)
You can open projects from QT Creator by selecting the .qbs file, edit, run etc.


## Create a new project
###QTCreator
The addon management in **QT Creator** is awesome (hat's off Arturo and team!), so all you have to do is add "ofxMSATensorFlow" under the addons section and it works. You can even do this while the project is open. This works on both **OSX** and **Linux**. 

###XCode
The **project generator** works nicely with the addon for **xcode** (details in addons_config.mk).

###Other
You need to include the library libtensorflow_cc.so for your plaatform and the header search folders listed in [addon_config.mk](https://github.com/memo/ofxMSATensorFlow/blob/master/addon_config.mk)

----

## Install TensorFlow for Python [Optional, but recommended]
The above is all you need to play with tensorflow inside openframeworks. However a good workflow is to install the Python version of tensorflow as well, create and train models in python, then load those models in openframeworks, feed it data and get the results.

Just follow the normal instructions here. 

You don't need to build from source, the PIP/VirtualEnv/Docker methods should all work. 

----
## Rebuild the library [Optional, not recommended, but you'll know if you need to]
The above just installs the python bindings for tensorflow, it has no effect on the addon. Alternatively, If you want to incorporate new updates in tensorflow code, or compile with different options (e.g. GPU, debug, release etc) you need to build tensorflow from source and create a new lib.

Follow the 'Installing from sources' instructions at 
https://www.tensorflow.org/versions/master/get_started/os_setup.html#installing-from-sources

**IMPORTANT**: However,  clone **my repo** for tensorflow. Or clone the original, but merge from mine (I have an updated build file with the C++ stuff). Remember to get the submodules. 

	git clone --recurse-submodules http://github.com/memo/tensorflow

or 

	git clone --recurse-submodules https://github.com/tensorflow/tensorflow
	git pull http://github.com/memo/tensorflow
	
**Once you've installed all of the dependencies and other instructions (e.g. if building with GPU), to build the C++ library go to the root of your tensorflow folder and type one of the following:**

	# for optimized (release) lib (~42MB)
	bazel build -c opt //tensorflow:libtensorflow_cc.so

	# for optimized (release) lib with GPU (linux only) (~140MB)
	# Make sure you've followed the CUDA + cuDNN setup instructions and run ./configure accordingly
	bazel build -c opt --config=cuda //tensorflow:libtensorflow_cc.so

	# for debug lib (~330MB)
	bazel build //tensorflow:libtensorflow_cc.so


Once it's finished, grab the file tensorflow/bazel-bin/tensorflow/libtensorflow.so and overwrite the one in ofxMSATensorFlow/libs/lib and anywhere else that you need it (e.g. the LD search path that you decided to put it in above)

### Copy headers
I provide all the nessecary headers in ofxMSATensorFlow/libs/tensorflow/include. If the tensorflow codebase hasn't changed, and you're only building for a new platform or options, then you probably won't need to update these. But if you've pulled an updated version of tensorflow and the code has changed, you will need to copy the new headers over. I've written a script for this and you can find it in ofxMSATensorFlow/scripts. Run the script to copy the files, or look at it - it's quite human readable hopefully - to see what is being copied over. 

Note, it's not just .hpp and .h that need to be copied. There are a lot of extensionless files being included too. So for now I'm copying the entire folders over, cpps, scripts, make files and all. The script file has an option to delete all .c .cpp .cxx .dat files etc after copying, but it's disabled by default for safety.

Also note that some of the headers needed are generated by the build processes mentioned above. So make sure you build (i.e. run bazel) before running the copy script. 


### Protobuf
Protobuf is notorious for causing pain as far as I can tell. It's a library for serialization, and generates headers based on message serialization structure defined in .proto files. It's a nice idea, but is incredibly sensistive to versions. I.e. it doesn't seem backwards compatible so if you're using a library compiled with one version of protobuf with headers generated from an ever so slightly different version, it'll fail. Tensorflow requires >v3. Public release is v2.6.1. So if you have that installed somewhere on your system it might break things. The instructions above should install v3+ (v3.0.0.a3 at time of writing) from source. But if you run into problems mentioning protobuf, it's probably a version conflict. The below should fix it, BUT it might break other software which requires older versions of protobuf. Running isolated environments would be your best bet in that case. JOY. 

#### Protobuf problems in Python

If you have problems in python regarding protobuf (e.g. when importing) try the below:

	sudo pip uninstall protobuf 
	sudo pip install protobuf==3.0.0a3 # or whatever the most recent bundled, but unreleased version is
	sudo pip uninstall tensorflow
	sudo pip install tensorflow-xxxxx.whl

Note. The name of the tensorflow-xxxxx.whl might be different on your system. Look for it with 

	ls -l /tmp/tensorflow_pkg


#### Protobuf problems in C++
In some cases, you might have problems in C++. If you have remnants of old protobuf headers somewhere in your header search paths, the compiler might load that instead of the v3+ ones and cry about it. In this case  I've had to install protobuf from source (not the python pip wheel, but the actual library in the system).

First remove all traces of protobuf installed via apt

	sudo apt-get purge libprotobuf-dev

Clone the protobuf repo to a *new* folder (don't build from the protobuf folder inside tensorflow, as it can mess things up). Somewhere other than your tensorflow folder:

	git clone https://github.com/google/protobuf

Go into the folder and type

	sudo apt-get install autoconf automake libtool curl # get dependencies
	./autogen.sh
	./configure
	make
	make check
	sudo make install
	sudo ldconfig # refresh shared library cache.

Instructions in the protobuf README https://github.com/google/protobuf/blob/master/src/README.md

# NOTES:
Here are some outstanding issues and misc notes.

## Jpegs don't load!
Perhaps the biggest issue is jpegs don't load for some reason. ofImage.load doesn't work with jpegs. FreeImage_Load fails. Neither does cvLoadImage (or cv::imread). If I don't use tensorflow, it works. If I use tensorflow, it doesn't. Just fails silently. Needs serious investigation. PNGs work fine. 

## No system dialogs
Whatever it is that implements ofSystemDialog, was compiled with protobuf 2.6.1 whereas tensorflow needs protobuf>3.0.0 (not released yet, using dev version). So can't use ofSystemDialogs. Also needs investigation. 

## QT Creator and Protobuf (probably linked to the above)
QT Creator requires libgtk-3-dev, which requires libprotobuf v2.6. So removing libprotobuf renders QT Creator useless. Installing QT Creator (or libgtk-3-dev) installs libprotobuf v2.6. I'm not enough of a linux expert to solve this problem. Will investigate. In the meantime, with QT Creator and libgtk-3-dev (and libprotobuf v2.6.1) installed it all works fine if you follow instructions above and install protobufv3 from source as well. However, there are some problems (like system dialogs not working mentioned above). Needs investigation. 

## X.h defines
X.h was #defining a bunch of common words like Success, Status, None etc to ints. Those are class names in tensorflow so it was causing the compilation to fail. So before including any tensorflow headers I #undef them (I do this in ofxMSATensorFlow.h, so you don't have to do it). BUT there might be side effects.


## Lib compilation options
#### Debug vs Release
The pre-compiled library I'm including is Release. Follow instructions above to build Debug version. Currently I'm not sure how to manage both release and debug libraries simultaneously in the project with the custom of.qbs template we're using. Usually you'd put -ltensorflow under release configuration options and -ltensorflowd under debug configuration options. But I couldn't find that in the way the openframeworks qbs template is setup. Needs investigation.

**Note**: You can run your ofApp in debug with the release lib and vice versa however:

* obviously you can't debug tensorflow code with the release lib (but you can debug your own code)
* certain operations cause it to crash (segmentation fault). The same operations work when you match debug lib to debug app and release lib to release app. But if you mismatch, it crashes. Not all operations, just some (I think it's the cc ops, needs more debugging). So if your app is constantly crashing and you're not sure why, check the debug/release settings vs the lib. 


## Eigen3
tensorflow uses Eigen in the backend for Tensor stuff. So feel free to use it. It's included in the addon. (And don't include eigen in your project separately)


## Trained variables not saved
Nothing to do with ofxMSATensorFlow, but a big gotcha for me was discovering that models saved from python don't save the trained parameters!
Python's *tf.train.write_graph* method saves the graph, but doesn't save the variable values (i.e. parameters) of the model. Which kind of defeats the purpose. Currently there's a few hack workarounds for this. I have these implented and commented in the MNIST examples.
