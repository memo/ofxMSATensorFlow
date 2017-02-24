
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
Simplest example possible. A very simple graph that multiples two numbers is built in python and saved. The openframeworks example loads the graph, and feeds it mouse coordinates. 100s of lines of code, just to build a simple multiplication function. 

### example-build-graph
Builds a simple graph from scratch directly in openframeworks using the C++ API without any python. Really not very exciting to look at, more of a syntax demo than anything. Based on https://www.tensorflow.org/api_guides/cc/guide

### example-char-rnn
Generative character based LSTM RNN demo, ala [Karpathy's char-rnn](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) and [Graves2013](https://arxiv.org/abs/1308.0850).
![](https://cloud.githubusercontent.com/assets/144230/23296346/74d8a194-fa6c-11e6-90c2-fb02084eb82b.png)
Models are trained and saved in python with [this code](https://github.com/memo/char-rnn-tensorflow) and loaded in openframeworks for prediction. I'm supplying a bunch of models (bible, cooking, erotic, linux, love songs, shakespeare, trump), and while the text is being generated character by character (at 60fps!) you can switch models in realtime mid-sentence or mid-word. (Drop more trained models into the folder and they'll be loaded too). Typing on the keyboard also primes the system, so it'll try and complete based on what you type. This is a simplified version of what I explain [here](https://vimeo.com/203485851), where models can be mixed as well. (Note, all models are trained really quickly with no hyperparameter search or cross validation, using default architecture of 2 layer LSTM of size 128 with no dropout or any other regularisation. So they're not great. A bit of hyperparameter tuning would give much better results - but note that would be done in python. The openframeworks code won't change at all, it'll just load the better model).

### example-mnist
MNIST (digit) clasffication with two different models - shallow and deep. Both models are built and trained in python (py src in bin/py folder). Openframeworks loads the trained models, allows you to draw with your mouse, and tries to classify your drawing. Toggle between the two models with the 'm' key.
![](https://cloud.githubusercontent.com/assets/144230/12665280/8fa4612a-c62e-11e5-950e-eaec14d4211d.png)

**Single layer softmax regression:** Very simple multinomial logistic regression. Quick'n'easy but not very good. Trains in seconds. Accuracy on test set ~90%. 
Implementation of https://www.tensorflow.org/versions/0.6.0/tutorials/mnist/beginners/index.html

**Deep(ish) Convolutional Neural Network:** Basic convolutional neural network. Very similar to LeNet. Conv layers, maxpools, RELU's etc. Slower and heavier than above, but much better. Trains in a few minutes (on CPU). Accuracy 99.2%
Implementation of https://www.tensorflow.org/versions/0.6.0/tutorials/mnist/pros/index.html#build-a-multilayer-convolutional-network

### example-inception3
Openframeworks implementation for image recognition using Google's 'Inception-v3' architecture network, pre-trained on ImageNet. Background info at https://www.tensorflow.org/versions/0.6.0/tutorials/image_recognition/index.html
![](https://cloud.githubusercontent.com/assets/144230/23235025/e88d8a40-f94b-11e6-9f3b-c5c65906c1a4.png)

### example-tests
Just some unit tests. Very boring for most humans. Possibly exciting for computers (or humans that get excited at the thought of computers going wrong).

