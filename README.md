
# ofxMSATensorFlow
This is an openframeworks addon for Google's data-flow graph based numerical computation / machine intelligence library TensorFlow https://www.tensorflow.org/.

TensorFlow is written in C/C++ with python bindings, and most of the documentation and examples are for python. This addon wraps the C/C++ backend with a number of examples. The basic idea is:

	1. build and train flow-graphs/models in python, C++ or any other language/platform with bindings
	2. Save the graphs/models to binary files
	3. Load the graphs/models in openframeworks, manipulate, feed it data, get results, play with them
You could potentially do steps 1-2 in openframeworks as well, but it seems a lot of that functionality is a lot more cumbersome in C++ compared to python. Also AFAIK the C++ API isn't as developed for training.

*(TBH Since training deep learning models takes so long and is very non-realtime, I don't think it makes too much sense to put yourself through the pain of implementing models in a syntactically tortuous language like C++, when C-backed, highly optimized, often GPU based python front ends are available for building and training models. However, once a model is trained, linking it to all kinds of other bits in realtime in an openframworks-like environment is where the fun's at!)*

**Note**: The pre-compiled library I provide is for **Linux only**, though building for OSX should be very simple (I just don't have a Mac right now). Windows might be a bit more of a pain since Bazel (the build platform) is *nix only - and would involve either porting Bazel, or rebuilding make/cmake files.
The project files for the examples are for **QTCreator**, so should work on all platforms out of the box? But anyways it's just one library and 3 header include files, so setting up other IDEs should be very simple. 

Since this is such an early version of the addon, I'll probably break backwards  compatibility with new updates. Sorry about that!

And there are a number of issues which I'll mention at the end. 

---
# Examples
I have a bunch more half-finished examples which I need to tidy up. In the meantime included examples are...

## example-basic
The hello world (no not MNIST, that comes next). Build a graph in python that multiplies two numbers. Load the graph in openframeworks and hey presto. 100s of lines of code, just to build a simple multiplication function. 

## example-mnist
MNIST clasffication with two different models - shallow and deep. Both models are built and trained in python (in bin/py folder). Loaded, manipulated and interacted with in openframeworks.
![](https://cloud.githubusercontent.com/assets/144230/12665280/8fa4612a-c62e-11e5-950e-eaec14d4211d.png)

####Single layer softmax regression: 
Very simple, quick'n'easy but not very good. Trains in seconds. Accuracy on validation ~90%. 
Implementation of https://www.tensorflow.org/versions/0.6.0/tutorials/mnist/beginners/index.html

####Deep(ish) Convolutional Neural Network:
Conv layers, maxpools, RELU's etc. Slower and heavier than above, but much better. Trains in a few minutes (on CPU). Accuracy 99.2%
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
Get openframeworks for linux (download or clone repo) http://openframeworks.cc/download/

Follow instructions on setting it up, dependencies, compiling etc. http://openframeworks.cc/setup/linux-install/

## Get QT Creator IDE
I've supplied project files for QT Creator IDE. So quickest way to get up and running is to use that (I've never used it before, but so far it looks pretty decent). Download and install the QT Creator IDE http://openframeworks.cc/setup/qtcreator/
It shouldn't be too hard to setup a new projects for other IDEs. More on this below. 


## Get ofxMSATensorFlow
Download or clone the repo ofxMSATensorFlow into your openframeworks/addons folder
https://github.com/memo/ofxMSATensorFlow

## Download binaries
Important! You need the precompiled lib, and data for the examples. You can find these in the Releases section. https://github.com/memo/ofxMSATensorFlow/releases


Copy the files to their corresponding folders. (e.g. from *downloaded/libs/lib/* to *ofxMSATensorFlow/libs/lib/*)

## Set your library folder
I made the library a shared library (.so) instead of static (.a) because it's huge! (340MB for debug).
It was easier this way, can think about alternatives for the future. 
So in the meantime, either copy the .so file to an existing location which is checked for .so's, or add the folder it's in to your system lib path. Explained here http://blog.andrewbeacock.com/2007/10/how-to-add-shared-libraries-to-linuxs.html
E.g. on ubuntu, in the terminal (CTRL+ALT+T):

	sudo gedit /etc/ld.so.conf.d/mylibs.conf

It will ask for your password and then open a (probably empty) text file. Write the folder containing your .so file. e.g.:

	/home/memo/DEV/tensorflow/bazel-bin/tensorflow
Save and close. Then in the terminal again type 

	sudo ldconfig



## THAT'S IT!!! (See my notes section at the end for caveats)
You can open projects from QT Creator by selecting the .qbs file, edit, run etc.


## Create a new project
The addon management in QT Creator is awesome (hat's off Arturo and team!), just edit the .qbs file and put in "ofxMSATensorFlow" under the addons section and it works. You can do this while the project is open.

Or there's the old school way:

	1. Duplicate an existing project folder, e.g. example-basic
	2. delete the .qbs.user file and the executable in the bin folder
 	3. rename the .qbs to whatever, e.g. myapp.qbs

----

## Install TensorFlow for Python [Optional, but recommended]
The above is all you need to play with tensorflow inside openframeworks. However the C++ API is quite limited. It's great for playing with pre-trained models, but AFAIK it's a bit limited (or PITA) for actually training models right now. So a good workflow is to install the Python version of tensorflow as well, create and train models in python, then load those models in openframeworks, feed it data and get the results.

Just follow the normal instructions here. 
https://www.tensorflow.org/versions/0.6.0/get_started/os_setup.html#download-and-setup
You don't need to build from source, the PIP/VirtualEnv/Docker methods should all work. 

----
## Build your own addon lib [Optional, not recommended, but you'll know if you need to]
The above just installs the python bindings for tensorflow, it has no effect on the addon.  Alternatively, If you want to incorporate new updates in tensorflow code, or compile with different options (e.g. GPU, debug, release etc) you need to build tensorflow from source and create a new lib.

### Install Bazel
Follow instructions at http://bazel.io/docs/install.html

### Clone the TensorFlow repo
**Important**: either clone from, or merge from http://github.com/memo/tensorflow (I have an updated build file with the C++ stuff). Remember to get the submodules. 

	git clone --recurse-submodules http://github.com/memo/tensorflow

or 

	git clone --recurse-submodules https://github.com/tensorflow/tensorflow
	git pull http://github.com/memo/tensorflow

### Install dependencies

	sudo apt-get install python-pip python-dev python-numpy swig 

*(Note: at time of writing, the pip that comes with apt-get on Ubuntu is really old and terrible v1.5.6. I suggest updating it immediately and restart your terminal:*)

	sudo pip install -U pip


### Build

In the root of the tensorflow folder run configuration script and follow instructions (you just need to answer two questions, location of your python (defaults to /usr/bin/python), and whether or not you're using GPU).
	
	./configure

On the tensorflow website 'build from sources section' there are instructions to build and install the python version. These aren't actually nessecary for our purposes. They will rebuild a python pipwheel and install that. If you already have a python version working and you're happy with that, and all you want to do is build the c++ lib with different compilation options etc, then you can skip this step. However if you are re-building the lib because there are significant changes in the code that is not yet reflected in the public release of the python version, then you should do these steps. 

	bazel build -c opt //tensorflow/tools/pip_package:build_pip_package
	
	# To build with GPU support:
	bazel build -c opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
	
	bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
	
	# The name of the .whl file will depend on your platform.
	pip install /tmp/tensorflow_pkg/tensorflow-0.6.0-py2-none-any.whl
*(Note, due to various reasons, I had to do sudo pip install for the last command)*
	

**This is the important step to rebuld the c++ lib. Go to the root of your tensorflow folder and type**

	# for optimized (release) lib (~42.MB)
	bazel build -c opt //tensorflow:libtensorflow.so

	# for debug lib (~330MB)
	bazel build //tensorflow:libtensorflow.so

Once it's finished, grab the file tensorflow/bazel-bin/tensorflow/libtensorflow.so and overwrite the one in ofxMSATensorFlow/libs/lib and anywhere else that you need it (e.g. the LD search path that you decided to put it in above)

### Copy headers
I provide all the nessecary headers in ofxMSATensorFlow/libs/tensorflow/include. If the tensorflow codebase hasn't changed, and you're only building for a new platform or options, then you probably won't need to update these. But if you've pulled an updated version of tensorflow and the code has changed, you will need to copy the new headers over. I've written a script for this and you can find it in ofxMSATensorFlow/scripts. Run the script to copy the files, or look at it - it's quite human readable hopefully - to see what is being copied over. 

Note, it's not just .hpp and .h that need to be copied. There are a lot of extensionless files being included too. So for now I'm copying the entire folders over, cpps, scripts, make files and all. I'm sure it can be pruned and there's a lot that isn't needed. 

Also note that some of the headers needed are generated by the build processes mentioned above. So make sure you build (i.e. run bazel) before running the copy script. 


### Protobuf
Is a PITA. Protobuf has caused endless pain for me on various projects. Usually due to version issues. Tensorflow requires >v3. Public release is v2.6.1. So if you have that installed somewhere on your system it might break things. The instructions above should install v3+ (v3.0.0.a3 at time of writing) from source. But if you run into problems mentioning protobuf, it's probably a version conflict.

#### Python

If you have problems in python regarding protobuf (e.g. when importing) try the below:

	sudo pip uninstall protobuf 
	sudo pip install protobuf==3.0.0a3
	sudo pip uninstall tensorflow
	sudo pip install tensorflow-0.6.0-py2-none-any.whl

Note. The name of the tensorflow-xxxx.whl might be different on your system. Look for it with 

	ls -l /tmp/tensorflow_pkg


#### C++
In some cases, you might have problems in C++. If you have remnants of old protobuf headers somewhere in your header search paths, the compiler might load that instead of the v3+ ones and cry about it. In this case  I've had to install protobuf from source (not the python pip wheel, but the actual library in the system).

First remove all traces of protobuf installed via apt

	sudo apt-get purge libprotobuf-dev

Clone the protobuf repo to a *new* folder (not the one inside tensorflow, as it can mess up things). Go into the folder and type

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


## No image load! - See UPDATE below
ofImage.load doesn't work for some reason. FreeImage_Load fails. Neither does cvLoadImage (or cv::imread). If I don't use tensorflow, it works. If I use tensorflow, it doesn't. Just fails silently. Needs serious investigation. (So for the inception demo, I had to test it by converting jpgs to raw data in gimp and load them via ofBuffers :/)

## No system dialogs - see UPDATE below
Whatever it is that implements ofSystemDialog, was compiled with protobuf 2.6.1 whereas tensorflow needs protobuf>3.0.0 (not released yet, using dev version). So can't use ofSystemDialogs. Also needs investigation. 

## X.h defines
X.h was #defining a bunch of common words like Success, Status, None etc to ints. Those are class names in tensorflow so it was causing the compilation to fail. So before including any tensorflow headers I #undef them (I do this in ofxMSATensorFlow.h, so you don't have to do it). BUT there might be side effects.



## Lib compilation options
#### Debug vs Release
The pre-compiled library I'm including is Release. Follow instructions above to build Debug version. Currently I'm not sure how to manage both release and debug libraries simultaneously in the project with the custom of.qbs template we're using. Usually you'd put -ltensorflow under release configuration options and -ltensorflowd under debug configuration options. But I couldn't find that in the way the openframeworks qbs template is setup. Needs investigation.

**Note**: You can run your ofApp in debug with the release lib and vice versa however:

* obviously you can't debug tensorflow code with the release lib (but you can debug your own code)
* certain operations cause it to crash (segmentation fault). The same operations work when you match debug lib to debug app and release lib to release app. But if you mismatch, it crashes. Not all operations, just some (I think it's the cc ops, needs more debugging). So if your app is constantly crashing and you're not sure why, check the debug/release settings vs the lib. 

#### GPU vs CPU
I built the lib for CPU only as GPU is a  bit more complex (requires CUDA and possibly CuDNN). Follow instructions above to rebuild lib if you want GPU support.

## Eigen3
tensorflow uses Eigen in the backend for Tensor stuff. So feel free to use it. It's included in the addon. (And don't include eigen in your project separately)



where ***addon_root*** is the full absolute path to the addon
e.g. /home/memo/DEV/of_v0.9.0_linux64_release/addons/ofxMSATensorFlow/

## Trained variables not saved
Nothing to do with ofxMSATensorFlow, but a big gotcha for me was discovering that models saved from python don't save the trained parameters!
Python's *tf.train.write_graph* method saves the graph, but doesn't save the variable values (i.e. parameters) of the model. Which kind of defeats the purpose. Currently there's a few hack workarounds for this. I have these implented and commented in the MNIST examples.

-----

#UPDATE
QT Creator requires libgtk-3-dev, which requires libprotobuf v2.6
So removing libprotobuf renders QT Creator useless.
Installing QT Creator (or libgtk-3-dev) installs libprotobuf v2.6
I'm not enough of a linux expert to solve this problem. Will investigate. 

In the meantime, with QT Creator and libgtk-3-dev (and libprotobuf v2.6.1) installed it all works fine if you follow instructions above and install protobufv3 from source as well. However, there are some problems (like system dialogs not working). Needs investigation. 
