/*
 * Very simple example demonstrating loading a graph and manipulating it
 * based on https://www.tensorflow.org/versions/master/api_docs/cc/ClassSession.html
 * and https://medium.com/jim-fleming/loading-a-tensorflow-graph-with-the-c-api-4caaff88463f#.9nywhxemo
 *
 * Python script constructs tensor flow graph which simply multiplies two numbers and exports binary model (see bin/py)
 *
 * openFrameworks code loads and processes pre-trained model (i.e. makes calculations/predictions)
 *
 */

#include "ofMain.h"
#include "ofxMSATensorFlow.h"

class ofApp : public ofBaseApp {
public:

    // shared pointer to tensorflow::Session
    msa::tf::Session_ptr session;


    //--------------------------------------------------------------
    void setup() {
        ofSetColor(255);
        ofBackground(0);
        ofSetVerticalSync(true);

        // Load graph (i.e. trained model) we exported from python, and initialize session
        session = msa::tf::create_session_with_graph("models/model.pb");
    }


    //--------------------------------------------------------------
    void draw() {
        stringstream message;

        if(session) {
            // inputs are linked to mouse position, normalized to 0..10
            float a = round(ofMap(ofGetMouseX(), 0, ofGetWidth(), 0, 10));
            float b = round(ofMap(ofGetMouseY(), 0, ofGetHeight(), 0, 10));

            // convert to tensors
            auto t_a = msa::tf::scalar_to_tensor(a);
            auto t_b = msa::tf::scalar_to_tensor(b);

            // output tensors will be stored in this
            vector<tensorflow::Tensor> outputs;

            // Run the graph, pass in
            // inputs to feed: array of dicts of { node name : tensor } )
            // outputs to fetch: array of node names
            // evaluates operation and return tensors in last parameter
            // The strings must match the name of the variable/node in the graph
            session->Run({ { "a", t_a }, { "b", t_b } }, { "c" }, {}, &outputs);

            // outputs is a vector of tensors, in our case we fetched only one tensor
            auto t_c = outputs[0];

            // get scalar value of tensor
            float c = msa::tf::tensor_to_scalar<float>(t_c);

            // Print the results
            message << "MOVE MOUSE!" << endl << endl;
            message << a << " (" << t_a.DebugString() << ")" << endl;
            message << " * " << endl;
            message << b << " (" << t_b.DebugString() << ")" << endl;
            message << " = " << endl;
            message << c << " (" << t_c.DebugString() << ")" << endl;
            message << endl;
            message << "all this madness, just to calculate a * b" << endl;

        } else {
            message << "Error during initialization, check console for details.";

        }

        ofDrawBitmapString(message.str(), 30, 30);
    }

};

//========================================================================
int main() {
    ofSetupOpenGL(1024, 768, OF_WINDOW);
    ofRunApp(new ofApp());
}
