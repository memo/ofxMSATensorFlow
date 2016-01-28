#pragma once

#include "ofMain.h"
#include "ofxMSATensorFlow.h"
#include "MousePainter.h"


class ofApp : public ofBaseApp{
public:

    // main interface to everything tensorflow
    ofxMSATensorFlow    msa_tf;

    // input tensors
    tensorflow::Tensor input_tensor; // will contain flattened bytes of an image

    // pixels to store input pixels. Creating these here so they aren't reallocated every frame
    ofPixels cpix;
    ofFloatPixels fpix;

    // output tensors
    // output_tensors[0] will contain 10 probabilities (one per class)
    vector<tensorflow::Tensor> output_tensors;

    // helper for painting with mouse, used as input image
    MousePainter mouse_painter;

    // will use for visualizing weights.
    // inner storage is for each node of layer, outer storage is for layers
    vector< vector< std::shared_ptr<ofFloatImage> > > weight_imgs;
//    vector< std::shared_ptr<ofFloatImage> > bias_imgs;

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
