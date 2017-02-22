/*
 * Simple example for constructing a flow graph in C++
 * based on https://www.tensorflow.org/api_guides/cc/guide
 *
 * Graph construction done directly in openFrameworks/C++
 * Check console for output
 *
 */


#include "ofMain.h"
#include "ofxMSATensorFlow.h"

class ofApp : public ofBaseApp {
public:

    //--------------------------------------------------------------
    void setup() {
        ofBackground(0);
        ofSetVerticalSync(true);

        ofLogNotice() << "Starting...";

        // create a root scope which will contain the graph and things
        auto root_scope = tensorflow::Scope::NewRootScope();

        // add a Matrix to the graph, store a reference to it in variable A
        auto A = tensorflow::ops::Const(root_scope, { {3.f, 2.f}, {-1.f, 0.f}});

        // add a Vector to the graph, store a reference to it in variable b
        auto b = tensorflow::ops::Const(root_scope, { {3.f, 5.f}});

        // add the operation A * b^T to the graph, store a reference to it in variable v
        auto v = tensorflow::ops::MatMul(root_scope.WithOpName("v"), A, b, tensorflow::ops::MatMul::TransposeB(true));

        // init the session TODO: check why this doesn't work.
        tensorflow::ClientSession session(root_scope);

        // run the operation 'v' and get the results in outputs
        std::vector<tensorflow::Tensor> outputs;
        TF_CHECK_OK(session.Run({v}, &outputs));

        ofLogVerbose() << outputs[0].matrix<float>();
    }

    //--------------------------------------------------------------
    void draw() {
        ofSetColor(255);
        ofDrawBitmapString("Check console for output", 30, 30);
    }

};

//========================================================================
int main() {
    ofSetupOpenGL(1024,768,OF_WINDOW);
    ofRunApp(new ofApp());
}
