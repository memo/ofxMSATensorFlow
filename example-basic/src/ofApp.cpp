/*
 * Very simple example demonstrating loading a graph and manipulating it
 * based on https://www.tensorflow.org/versions/master/api_docs/cc/ClassSession.html
 * and https://medium.com/jim-fleming/loading-a-tensorflow-graph-with-the-c-api-4caaff88463f#.9nywhxemo
 *
 * Python script constructs tensor flow graph which simply multiplies two numbers, exports binary model (see bin/py)
 *
 * openFrameworks code loads and processes pre-trained model (i.e. makes calculations/predictions)
 *
 */

#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup(){
    ofSetColor(255);
    ofBackground(0);
    ofSetVerticalSync(true);

    // Initialize tensorflow session, return if error
    if( !msa_tf.setup() ) return;

    // Load graph (i.e. trained model) we exported from python, add to session, return if error
    if( !msa_tf.loadGraph("models/model.pb") ) return;

    // initialize input tensor dimensions
    // (not sure what the best way to do this was as there isn't an 'init' method, just a constructor)
    a = tensorflow::Tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape());
    b = tensorflow::Tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape());
}

//--------------------------------------------------------------
void ofApp::update(){

}

//--------------------------------------------------------------
void ofApp::draw(){
    stringstream message;

    if(msa_tf.isReady()) {
        // inputs are linked to mouse position, normalized to 0..10
        a.scalar<float>()() = round(ofMap(ofGetMouseX(), 0, ofGetWidth(), 0, 10));
        b.scalar<float>()() = round(ofMap(ofGetMouseY(), 0, ofGetHeight(), 0, 10));

        // Collect inputs into a vector
        // IMPORTANT: the string must match the name of the variable/node in the graph
        vector<pair<string, tensorflow::Tensor>> inputs = {
            { "a", a },
            { "b", b },
        };

        // desired outputs which we want processed and returned from the graph
        // IMPORTANT: the string must match the name of the variable/node in the graph
        vector<string> output_names = { "c" };

        // Run the graph, pass in our inputs and desired outputs, evaluate operation and return
        if( !msa_tf.run(inputs, output_names, {}, &outputs) ) message << "Error during running. Check console for details." << endl;

        // outputs is a vector of tensors, we're interested in only the first tensor
        auto &c = outputs[0];

        // get scalar values of each tensor (since they're 1D and single element it's easy)
        float val_a = a.scalar<float>()();
        float val_b = b.scalar<float>()();
        float val_c = c.scalar<float>()();

        // Print the results
        message << "MOVE MOUSE!" << endl << endl;
        message << val_a << " (" << a.DebugString() << ")" << endl;
        message << " * " << endl;
        message << val_b << " (" << b.DebugString() << ")" << endl;
        message << " = " << endl;
        message << val_c << " (" << c.DebugString() << ")" << endl;
        message << endl;
        message << "all this madness, just to calculate a * b" << endl;

    } else {
        message << "Error during initialization, check console for details.";

    }

    ofDrawBitmapString(message.str(), 30, 30);
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){

}
