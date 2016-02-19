/*
 * Simple example for constructing a flow graph in C++
 * based on https://github.com/tensorflow/tensorflow/blob/master/tensorflow/cc/tutorials/example_trainer.cc
 *
 * Training (i.e. graph construction) done directly in openFrameworks/C++
 * Check console for output
 *
 */


#include "ofMain.h"
#include "ofxMSATensorFlow.h"


using namespace tensorflow;
using tensorflow::string;
using tensorflow::int32;


struct Options {
    //    int num_concurrent_sessions = 10;  // The number of concurrent sessions
    //    int num_concurrent_steps = 10;     // The number of concurrent steps
    int num_iterations = 100;          // Each step repeats this many times
    bool use_gpu = false;              // Whether to use gpu in the training
};


// Create and return Graph
// A = [3 2; -1 0]; x = rand(2, 1);
// We want to compute the largest eigenvalue for A.
// repeat x = y / y.norm(); y = A * x; end
void createGraphDef(tensorflow::GraphDef &g) {
    GraphDefBuilder b;
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)
    // Store rows [3, 2] and [-1, 0] in row major format.
    Node* a = Const({3.f, 2.f, -1.f, 0.f}, {2, 2}, b.opts());

    // x is from the feed.
    Node* x = Const({0.f}, {2, 1}, b.opts().WithName("x"));

    // y = A * x
    Node* y = MatMul(a, x, b.opts().WithName("y"));

    // y2 = y.^2
    Node* y2 = Square(y, b.opts());

    // y2_sum = sum(y2)
    Node* y2_sum = Sum(y2, Const(0, b.opts()), b.opts());

    // y_norm = sqrt(y2_sum)
    Node* y_norm = Sqrt(y2_sum, b.opts());

    // y_normalized = y ./ y_norm
    Div(y, y_norm, b.opts().WithName("y_normalized"));

    TF_CHECK_OK(b.ToGraphDef(&g));
}



string DebugString(const Tensor& x, const Tensor& y) {
    CHECK_EQ(x.NumElements(), 2);
    CHECK_EQ(y.NumElements(), 2);
    auto x_flat = x.flat<float>();
    auto y_flat = y.flat<float>();
    const float lambda = y_flat(0) / x_flat(0);
    return strings::Printf("lambda = %8.6f x = [%8.6f %8.6f] y = [%8.6f %8.6f]", lambda, x_flat(0), x_flat(1), y_flat(0), y_flat(1));
}


class ofApp : public ofBaseApp{
public:

    // main interface to everything tensorflow
    msa::tf::ofxMSATensorFlow    msa_tf;

    //--------------------------------------------------------------
    void setup(){
        ofBackground(0);
        ofSetVerticalSync(true);

        ofLogNotice() << "Starting...";

        // Initialize tensorflow session, return if error
        if( !msa_tf.setup() ) return;

        createGraphDef(msa_tf.graph());

        msa_tf.createSessionWithGraph();

        Options opts;

        // Randomly initialize the input.
        Tensor x(DT_FLOAT, TensorShape({2, 1}));
        x.flat<float>().setRandom();

        // Iterations.
        std::vector<Tensor> outputs;
        for (int iter = 0; iter < opts.num_iterations; ++iter) {
            outputs.clear();
            msa_tf.run({{"x", x}}, {"y:0", "y_normalized:0"}, {}, &outputs);
            CHECK_EQ(2, outputs.size());

            const Tensor& y = outputs[0];
            const Tensor& y_norm = outputs[1];
            // Print out lambda, x, and y.
            ofLogNotice() << iter << ": " << DebugString(x, y);
            // Copies y_normalized to x.
            x = y_norm;
        }

    }

    //--------------------------------------------------------------
    void draw(){
        ofSetColor(255);
        ofDrawBitmapString("Check console for output", 30, 30);
    }

};

//========================================================================
int main( ){
    ofSetupOpenGL(1024,768,OF_WINDOW);			// <-------- setup the GL context

    // this kicks off the running of my app
    // can be OF_WINDOW or OF_FULLSCREEN
    // pass in width and height too:
    ofRunApp(new ofApp());

}
