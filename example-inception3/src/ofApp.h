#pragma once

#include "ofMain.h"
#include "ofxMSATensorFlow.h"


class ofApp : public ofBaseApp{
public:

    // main interface to everything tensorflow
    ofxMSATensorFlow    msa_tf;

    // Tensor to hold input image which is fed into the network
    tensorflow::Tensor image_tensor;

    // vector of Tensors to hold data coming back from the network
    // (it's a vector of Tensors, because that's how the API works)
    vector<tensorflow::Tensor> output_tensors;

    // for webcam input
    shared_ptr<ofVideoGrabber> video_grabber;

    // contains input image to classify
    ofImage input_image;

    // normalized float version of input image
    ofFloatImage processed_image;

    // contains all labels
    vector<string> labels;

    // folder of images to classify
    ofDirectory image_dir;

    // contains classification information from last classification attempt
    vector<int> top_label_indices;
    vector<float> top_scores;

    // load next image in folder image_dir
    void loadNextImage();

    // run classification model on pixels
    void classify(ofPixels &pix);

    void setup();
    void update();
    void draw();
    void keyPressed(int key);
    void dragEvent(ofDragInfo dragInfo);
};
