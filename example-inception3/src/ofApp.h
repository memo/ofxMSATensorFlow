#pragma once

#include "ofMain.h"
#include "ofxMSATensorFlow.h"
#include "MousePainter.h"


class ofApp : public ofBaseApp{
public:

    // main interface to everything tensorflow
    ofxMSATensorFlow    msa_tf;

    // input tensors
    tensorflow::Tensor x_inputs; // will contain flattened bytes of an image

    // output tensors
    vector<tensorflow::Tensor> outputs;

    // for webcam input
    shared_ptr<ofVideoGrabber> video_grabber;

    // contains input image resized to correct dimensions for model
    ofImage resized_img;

    // processed float image
    ofFloatImage processed_img;

    // helper for painting with mouse, used as input image
    MousePainter mouse_painter;

    // contains all labels
    std::vector<string> labels;

    void loadNextImage();

    void runModel();

    void setup();
    void update();
    void draw();

    void keyPressed(int key);
    void keyReleased(int key);
    void mouseMoved(int x, int y );
    void mouseDragged(int x, int y, int button);
    void mousePressed(int x, int y, int button);
    void mouseReleased(int x, int y, int button);
    void mouseEntered(int x, int y);
    void mouseExited(int x, int y);
    void windowResized(int w, int h);
    void dragEvent(ofDragInfo dragInfo);
    void gotMessage(ofMessage msg);
};
