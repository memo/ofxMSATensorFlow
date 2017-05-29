
# ofxMSATensorFlow
[OpenFrameworks](http://openframeworks.cc) addon for Google's graph based machine intelligence / deep learning library [TensorFlow](https://www.tensorflow.org).

This update includes the newly released **TensorFlow r1.1** and has been tested with **openFrameworks 0.9.8**.

I provide precompiled libraries for **Linux** and **OSX** (though OSX might lag a little bit behind as I don't have regular access). For linux there are both **GPU** and **CPU**-only libs, while OSX is **CPU**-only. I haven't touched Windows yet as building from sources is 'experimental' (and doing Linux and OSX was painful enough).

You can find instructions and more information in the **[wiki](https://github.com/memo/ofxMSATensorFlow/wiki)**, particularly for **[Getting Started](https://github.com/memo/ofxMSATensorFlow/wiki/Getting-started)**.

---

TensorFlow is written in C/C++ with python bindings, and most of the documentation and examples are for python. This addon wraps the C/C++ backend (and a little bit of the new C++ FrontEnd) with a number of examples. The basic idea is:

1. **Build and train graphs** (i.e. 'networks', 'models') mostly in python (possibly Java, C++ or any other language/platform with tensorflow bindings)
2. **Save the trained models** to binary files
3. **Load the trained models in openframeworks**, feed them data, manipulate, get results, play, and connect them to the ofUniverse

You could potentially do steps 1-2 in openframeworks as well, but the python API is more user-friendly for building graphs and training. 

---
## Examples
The examples are quite minimal and shouldn't be considered comprehensive tensorflow tutorials. They demonstrate *loading and manipulating different types of tensorflow models in openFrameworks*. E.g.

* for **image classification** see *example-mnist* or *example-inception3*
* for **sequence generation of *discrete* data** such as text (with **stateful LSTM/RNN**, where LSTM state is retrieved and passed back in at every time-step) see *example-char-rnn*
* for **sequence generation of *continuous* data** such as handwriting (with **Recurrent Mixture Density Networks**) see *example-handwriting-rnn*
* for **image generation** (with **Conditional Generative Adversarial Networks**) see *example-pix2pix* or *example-pix2pix-webcam*
* for **constructing graphs in C++** see *example-build-graph*
* for the **most basic example** of loading a model, feeding it data and fetching the results, see *example-basic*

Potentially you could load any pretrained model in openframeworks and manipulate. E.g. checkout Parag's [tutorials](https://github.com/pkmital/tensorflow_tutorials) and [Kadenze course](https://www.kadenze.com/courses/creative-applications-of-deep-learning-with-tensorflow-iv/info). There's info in the wiki on [how to do export and distribute models](https://github.com/memo/ofxMSATensorFlow/wiki/Loading-and-using-trained-tensorflow-models-in-openFrameworks).

---

*(Note: the animations below are animated-gifs, hence the low color count, dithering, and low framerate)*

### example-pix2pix-webcam
![example-pix2pix-webcam_250x_short_800px_10fps](https://cloud.githubusercontent.com/assets/144230/26558171/ab2ebb3a-449e-11e7-92d9-30db72e6daa0.gif)

Same as pix2pix-example with the addition of live webcam input. See description of pix2pix-example for more info on pix2pix and the models I provide. I'm using a very simple and ghetto method of transforming the webcam input into the desired colour palette before feeding into the model. See the code for more info on this.

---

### example-pix2pix
![example-pix2pix_400x2x](https://cloud.githubusercontent.com/assets/144230/26264408/a4700456-3cd4-11e7-8b2c-632f99acac28.gif)

pix2pix (Image-to-Image Translation with Conditional Adversarial Nets). An accessible explanation can be found [here](https://phillipi.github.io/pix2pix/) and [here](https://affinelayer.com/pix2pix/). The network basically learns to map from one image to another. E.g. in the example you draw in the left viewport, and it generates the image in the right viewport. I'm supplying three pretrained models from the original paper: cityscapes, building facades, and maps. And a model I trained on [150 art collections from around the world](https://commons.wikimedia.org/wiki/Category:Google_Art_Project_works_by_collection). Models are trained and saved in python with [this code](https://github.com/memo/pix2pix-tensorflow) (which is based on [this](https://github.com/affinelayer/pix2pix-tensorflow) tensorflow implementation, which is based on the original [torch implementation](https://phillipi.github.io/pix2pix/)), and loaded in openframeworks for prediction. 

---

### example-handwriting-rnn
![](https://cloud.githubusercontent.com/assets/144230/23376150/08a3f866-fd23-11e6-9d9f-45738b1e9b2e.gif)

Generative handwriting with Long Short-Term Memory (LSTM) Recurrent Mixture Density Network (RMDN), ala [Graves2013](https://arxiv.org/abs/1308.0850). Brilliant tutorial on inner workings [here](http://blog.otoro.net/2015/12/12/handwriting-generation-demo-in-tensorflow/), which also provides the base for the training code (also see javscript port and tutorial [here](http://distill.pub/2016/handwriting/)).  Models are trained and saved in python with [this code](https://github.com/memo/write-rnn-tensorflow), and loaded in openframeworks for prediction. Given a sequence of points, the model predicts the position for the next point and pen-up probability. I'm supplying a model pretrained on the [IAM online handwriting dataset](http://www.fki.inf.unibe.ch/databases/iam-on-line-handwriting-database). Note that this demo does not do handwriting *synthesis*, i.e. text to handwriting ala [Graves' original demo](https://www.cs.toronto.edu/~graves/handwriting.html). It just does *asemic* handwriting, producing squiggles that are statistically similar to the training data, e.g. same kinds of slants, curvatures, sharpnesses etc., but not nessecarily legible. There is an implementation (and great tutorial) of *synthesis* using attention [here](https://greydanus.github.io/2016/08/21/handwriting/), which I am also currently converting to work in openframeworks. This attention-based synthesis implementation is also based on [Graves2013](https://arxiv.org/abs/1308.0850), which I highly recommend to anyone really interested in understanding generative RNNs.

---

### example-char-rnn
![](https://cloud.githubusercontent.com/assets/144230/23296346/74d8a194-fa6c-11e6-90c2-fb02084eb82b.png)

Generative character based Long Short-Term Memory (LSTM) Recurrent Neural Network (RNN) demo, ala [Karpathy's char-rnn](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) and [Graves2013](https://arxiv.org/abs/1308.0850).
Models are trained and saved in python with [this code](https://github.com/memo/char-rnn-tensorflow) and loaded in openframeworks for prediction. I'm supplying a bunch of models (bible, cooking, erotic, linux, love songs, shakespeare, trump), and while the text is being generated character by character (at 60fps!) you can switch models in realtime mid-sentence or mid-word. (Drop more trained models into the folder and they'll be loaded too). Typing on the keyboard also primes the system, so it'll try and complete based on what you type. This is a simplified version of what I explain [here](https://vimeo.com/203485851), where models can be mixed as well. (Note, all models are trained really quickly with no hyperparameter search or cross validation, using default architecture of 2 layer LSTM of size 128 with no dropout or any other regularisation. So they're not great. A bit of hyperparameter tuning would give much better results - but note that would be done in python. The openframeworks code won't change at all, it'll just load the better model).

---

### example-mnist
![](https://cloud.githubusercontent.com/assets/144230/12665280/8fa4612a-c62e-11e5-950e-eaec14d4211d.png)

MNIST (digit) clasffication with two different models - shallow and deep. Both models are built and trained in python (py src in bin/py folder). Openframeworks loads the trained models, allows you to draw with your mouse, and tries to classify your drawing. Toggle between the two models with the 'm' key.

**Single layer softmax regression:** Very simple multinomial logistic regression. Quick'n'easy but not very good. Trains in seconds. Accuracy on test set ~90%. 
Implementation of https://www.tensorflow.org/versions/0.6.0/tutorials/mnist/beginners/index.html

**Deep(ish) Convolutional Neural Network:** Basic convolutional neural network. Very similar to LeNet. Conv layers, maxpools, RELU's etc. Slower and heavier than above, but much better. Trains in a few minutes (on CPU). Accuracy 99.2%
Implementation of https://www.tensorflow.org/versions/0.6.0/tutorials/mnist/pros/index.html#build-a-multilayer-convolutional-network

---

### example-inception3
![](https://cloud.githubusercontent.com/assets/144230/23235025/e88d8a40-f94b-11e6-9f3b-c5c65906c1a4.png)

Openframeworks implementation for image recognition using Google's 'Inception-v3' architecture network, pre-trained on ImageNet. Background info at https://www.tensorflow.org/versions/0.6.0/tutorials/image_recognition/index.html

---

### example-tests
Just some unit tests. Very boring for most humans. Possibly exciting for computers (or humans that get excited at the thought of computers going wrong).

---

### example-basic
Simplest example possible. A very simple graph that multiples two numbers is built in python and saved. The openframeworks example loads the graph, and feeds it mouse coordinates. 100s of lines of code, just to build a simple multiplication function. 

---

### example-build-graph
Builds a simple graph from scratch directly in openframeworks using the C++ API without any python. Really not very exciting to look at, more of a syntax demo than anything. Based on https://www.tensorflow.org/api_guides/cc/guide
