
# ofxMSATensorFlow
[OpenFrameworks](http://openframeworks.cc) addon for Google's graph based numerical computation, machine intelligence, deep learning library [TensorFlow](https://www.tensorflow.org).

This update includes the newly released **TensorFlow r1.0**, and might have broken backwards compatibility. Sorry about that (tbh tensorflow has been breaking backwards compatibility with every update!). Hopefully from now on the API should be a bit more stable. 

Tested on **openFrameworks 0.9.8**.

I provide precompiled libraries for **Linux** and **OSX**. For linux there are both **GPU** and **CPU**-only libs, OSX is **CPU**-only (I don't have a Mac with NVidia). I haven't touched Windows yet as building from sources is 'experimental' (and doing Linux and OSX was painful enough).

You can find instructions and more information in the **[wiki](https://github.com/memo/ofxMSATensorFlow/wiki)**, particularly for **[Getting Started](https://github.com/memo/ofxMSATensorFlow/wiki/Getting-started)**.

---

TensorFlow is written in C/C++ with python bindings, and most of the documentation and examples are for python. This addon wraps the C/C++ backend (and a little bit of the new C++ FrontEnd) with a number of examples. The basic idea is:

1. Build graphs and/or train models in python, Java, C++ or any other language/platform with tensorflow bindings
2. Save the graphs or trained models to binary files
3. Load the graphs or trained models in openframeworks, feed them data, manipulate, get results, play, and connect them to the ofUniverse

You could potentially do steps 1-2 in openframeworks as well, but the python API is a bit more user-friendly for building graphs and training. 

---
## Examples
The examples are quite basic. They shouldn't be considered *tensorflow* examples or tutorials, but they mainly just demonstrate loading and manipulating of tensorflow models in openFrameworks. I really need to include more, but do checkout Parag's [tutorials](https://github.com/pkmital/tensorflow_tutorials) and [Kadenze course](https://www.kadenze.com/courses/creative-applications-of-deep-learning-with-tensorflow-iv/info) (both for tensorflow python). Building and training those models in python, and then playing with them in openframeworks should be relatively straight forward. 


### example-basic
The hello world (no not MNIST, that comes next). Build a very simple graph in python that multiplies two numbers. Load the graph in openframeworks and hey presto. 100s of lines of code, just to build a simple multiplication function. 

### example-mnist
MNIST clasffication with two different models - shallow and deep. Both models are built and trained in python (in bin/py folder). Loaded, manipulated and interacted with in openframeworks. Toggle between the two models with the 'm' key.
![](https://cloud.githubusercontent.com/assets/144230/12665280/8fa4612a-c62e-11e5-950e-eaec14d4211d.png)

**Single layer softmax regression:** Very simple multinomial logistic regression. Quick'n'easy but not very good. Trains in seconds. Accuracy on test set ~90%. 
Implementation of https://www.tensorflow.org/versions/0.6.0/tutorials/mnist/beginners/index.html

**Deep(ish) Convolutional Neural Network:** Basic convolutional neural network. Very similar to LeNet. Conv layers, maxpools, RELU's etc. Slower and heavier than above, but much better. Trains in a few minutes (on CPU). Accuracy 99.2%
Implementation of https://www.tensorflow.org/versions/0.6.0/tutorials/mnist/pros/index.html#build-a-multilayer-convolutional-network


### example-inception3
openframeworks implementation for image recognition using Google's 'Inception-v3' architecture network, pre-trained on ImageNet. Background info at https://www.tensorflow.org/versions/0.6.0/tutorials/image_recognition/index.html
![](https://cloud.githubusercontent.com/assets/144230/23235025/e88d8a40-f94b-11e6-9f3b-c5c65906c1a4.png)



### example-build-graph
Builds a simple flow graph in directly in openframeworks/C++ without using any python. (really not very exciting to look at. more of a syntax demo than anything). Based on https://www.tensorflow.org/api_guides/cc/guide


### example-tests
Just some unit tests. Very boring for most humans. Possibly exciting for computers (or humans that get excited at the thought of computers going wrong).

