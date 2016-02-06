/*
 * Two different MNIST classification examples
 * - Very simple softmax (and not very good) based on https://www.tensorflow.org/versions/master/tutorials/mnist/beginners/index.html
 * - Deep (better, but more code to follow) based on https://www.tensorflow.org/versions/master/tutorials/mnist/pros/index.html
 *
 * Python script downloads training data, constructs tensor flow graph, trains and exports binary model file (see bin/py)
 *
 * openFrameworks code loads and processes pre-trained model (i.e. makes calculations/predictions)
 *
 * uncomment GO_DEEP for the deep model below
 * (note, 99.999% of the openframeworks code doesn't care if the model is deep or not,
 * it just feeds data to the model (trained in python), and gets the results
 *
 */


#include "ofMain.h"
#include "ofxMSATensorFlow.h"
#include "MousePainter.h"


// uncomment this to switch to the deep model
#define GO_DEEP


// input image dimensions dictated by trained model
#define kInputWidth     28
#define kInputHeight    28
#define kInputSize      (kInputWidth * kInputHeight)

// output dimensions dictated by trained model
#define kOutputSize      10

// model file to load, depends on whether we're going deep or not
#ifdef GO_DEEP
#define kModelPath      "model-deep/mnist-deep.pb"
#else
#define kModelPath      "model-simple/mnist-simple.pb"
#endif


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


    //--------------------------------------------------------------
    void setup(){
        ofLogNotice() << "Initializing... ";
        ofBackground(50);
        ofSetVerticalSync(true);

        mouse_painter.setup(256);

        // Initialize tensorflow session, return if error
        if( !msa_tf.setup() ) return;

        // Load graph (i.e. trained model) we exported from python, add to session, return if error
        if( !msa_tf.loadGraph(kModelPath) ) return;


        // initialize input tensor dimensions
        // (not sure what the best way to do this was as there isn't an 'init' method, just a constructor)
        input_tensor = tensorflow::Tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({ 1, kInputSize }));

        // MEGA UGLY HACK ALERT
        // graphs exported from python don't store values for trained variables (i.e. parameters)
        // so in python we need to add the values back to the graph as 'constants'
        // and bang them here to push values to parameters
        // more here: https://stackoverflow.com/questions/34343259/is-there-an-example-on-how-to-generate-protobuf-files-holding-trained-tensorflow/34343517
        {
            std::vector<string> names;
            int node_count = msa_tf.graph().node_size();
            ofLogNotice() << node_count << " nodes in graph";

            // iterate all nodes
            for(int i=0; i<node_count; i++) {
                auto n = msa_tf.graph().node(i);
                ofLogNotice() << i << ":" << n.name();

                // if name contains var_hack, add to vector
                if(n.name().find("_VARHACK") != std::string::npos) {
                    names.push_back(n.name());
                    ofLogNotice() << "......bang";
                }
            }
            // run the network inputting the names of the constant variables we want to run
            if( !msa_tf.run({}, names, {}, &output_tensors) ) ofLogError() << "Error running network for weights and biases variable hack";
        }

        bool do_visualize_nodes = true;
        if(do_visualize_nodes) {

            // first find the layer names that have 'VIZ' in them
            // (I could have done this in the loop above, but I like to keep it modular and separate)
            std::vector<string> names;
            int node_count = msa_tf.graph().node_size();

            // iterate all nodes
            for(int i=0; i<node_count; i++) {
                auto n = msa_tf.graph().node(i);
                if(n.name().find("VIZ_VARHACK") != std::string::npos) {
                    names.push_back(n.name());
                }
            }

            // lets get the weights from the network to visualize them in images
            // run the network and ask for nodes with the names selected above
            if( !msa_tf.run({}, names, {}, &output_tensors)) ofLogError() << "Error running network to get viz layers";


            int nlayers = output_tensors.size();    // number of layers in network

            weight_imgs.resize(nlayers);

            for(int l=0; l<nlayers; l++) {
                // weights matrix is a bit awkward
                // each column contains flattened weights for each pixel of the input image for a particular digit
                // i.e. 10 columns (one per digit) and 784 rows (one for each pixel of the input image)
                // we need to transpose the weights matrix to easily get sections of it out, this is easy as an image
                ofFloatPixels weights_pix_full;  // rows: weights for each digit (10), col: weights for each pixel (784)
                ofxMSATensorFlow::tensorToPixels(output_tensors[l], weights_pix_full, false, "10");
                weights_pix_full.rotate90(1);   // now rows: weights for each pixel, cols: weights for each digit
                weights_pix_full.mirror(false, true);

                int nunits = weights_pix_full.getHeight();  // number of units in layer
                int npixels = weights_pix_full.getWidth();    // number of pixels per unit
                int img_size = sqrt(npixels);                 // size of image (i.e. sqrt of above)
                weight_imgs[l].resize(nunits);

                ofFloatImage timg; // temp single channel image
                for(int i=0; i<nunits; i++) {
                    weight_imgs[l][i] = make_shared<ofFloatImage>();

                    // get data from full weights matrix into a single channel image
                    int row_offset = i * npixels;
                    timg.setFromPixels(weights_pix_full.getData() + row_offset, img_size, img_size, OF_IMAGE_COLOR);

                    // convert single channel image into rgb (R showing -ve weights, B showing +ve weights)
                    float scaler = nlayers * nunits * 0.1; // arbitrary scaler to work with both shallow and deep model
                    ofxMSATensorFlow::grayToColor(timg, *weight_imgs[l][i], scaler);
                }
            }
        }


        ofLogNotice() << "Init successfull";
    }

    //--------------------------------------------------------------
    void update(){
        if(msa_tf.isReady()) {
            // get pixels from mousepainter and format correctly for input
            cpix = mouse_painter.get_pixels();
            cpix.resize(kInputWidth, kInputHeight); // resize to input dimensions
            cpix.setImageType(OF_IMAGE_GRAYSCALE);  // convert to single channel

            // convert from unsigned char to float, this also normalizes by 1/255
            fpix = cpix;

            // copy data from formatted pixels into tensor
            ofxMSATensorFlow::pixelsToTensor(fpix, input_tensor);


            // Collect inputs into a vector
            // IMPORTANT: the string must match the name of the variable/node in the graph
            vector<pair<string, tensorflow::Tensor>> inputs = { { "x_inputs", input_tensor } };

#ifdef GO_DEEP
            // set dropout probability to 1, not sure if this is the best way or if it can be disabled before saving the model
            tensorflow::Tensor keep_prob(tensorflow::DT_FLOAT, tensorflow::TensorShape());
            keep_prob.scalar<float>()() = 1.0f;
            inputs.push_back({"keep_prob", keep_prob});
#endif
            // Run the graph, pass in our inputs and desired outputs, evaluate operation and return
            // IMPORTANT: the string must match the names of the variables/nodes in the graph

            if( !msa_tf.run(inputs, { "y_outputs" }, {}, &output_tensors) )
                ofLogError() << "Error during running. Check console for details." << endl;
        }
    }

    //--------------------------------------------------------------
    void draw(){
        // draw mouse painter
        mouse_painter.draw();

        stringstream str_outputs;

        // will contain outputs of network
        vector<float> outputs;

        if(msa_tf.isReady() && !output_tensors.empty() && output_tensors[0].IsInitialized()) {

            // DRAW LAYER PARAMETERS
            float cur_y = mouse_painter.getHeight() + 10;
            ofSetColor(255);
            int nlayers = weight_imgs.size();
            for(int l=nlayers-1; l>=0; l--) {
                int nnodes = weight_imgs[l].size();
                //nnodes = min(nnodes, 32);   // don't draw all 128, doesn't fit on screen!
                float s = ofGetWidth() / (float)nnodes;

                for(int i=0; i<nnodes; i++) {
                    weight_imgs[l][i]->draw(i*s, cur_y, s*0.8, s*0.8);
                }
                cur_y += s;
            }

            // DRAW OUTPUT PROBABILITY BARS
            // copy from tensor to a vector using Mega Super Awesome convenience methods, because it's easy to deal with :)
            // output_tensors[0] contains the main outputs tensor
            // NB if the data was MASSIVE it would be faster to access directly from within the tensor
            ofxMSATensorFlow::tensorToVector(output_tensors[0], outputs);

            float box_spacing = ofGetWidth() / kOutputSize;
            float box_width = box_spacing * 0.8;

            for(int i=0; i<kOutputSize; i++) {
                float p = outputs[i]; // probability of this label

                // draw probability bar
                float h = (ofGetHeight() - cur_y) * p;
                float x = ofMap(i, 0, kOutputSize-1, 0, ofGetWidth() - box_spacing);
                x += (box_spacing - box_width)/2;

                ofSetColor(ofLerp(50.0, 255.0, p), ofLerp(100.0, 0.0, p), ofLerp(150.0, 0.0, p));
                ofDrawRectangle(x, ofGetHeight(), box_width, -h);

                str_outputs << ofToString(outputs[i], 3) << " ";

                // draw text
                ofDrawBitmapString(ofToString(i) + ": " + ofToString(p, 2), x, ofGetHeight() - h - 10);
            }


        }

        stringstream str;
        str << kModelPath << endl;
        str << "Outputs: " << str_outputs.str() << endl;
        str << endl;
        str << "Paint in the box" << endl;
        str << "Rightclick to erase" << endl;
        str << "'c' to clear" << endl;
        str << endl;
        str << "fps: " << ofToString(ofGetFrameRate(), 2) << endl;

        ofSetColor(255);
        ofDrawBitmapString(str.str(), mouse_painter.getWidth() + 20, 30);
    }

    //--------------------------------------------------------------
    void keyPressed(int key){
        switch(key) {
        case 'c': mouse_painter.clear(); break;
        }
    }

    //--------------------------------------------------------------
    void mouseDragged(int x, int y, int button){
        mouse_painter.penDrag(ofVec2f(x, y), button==2);
    }

    //--------------------------------------------------------------
    void mousePressed(int x, int y, int button){
        mouse_painter.penDown(ofVec2f(x, y), button==2);
    }

    //--------------------------------------------------------------
    void mouseReleased(int x, int y, int button){
        mouse_painter.penUp();
    }
};


//========================================================================
int main( ){
    ofSetupOpenGL(1600, 800, OF_WINDOW);			// <-------- setup the GL context

    // this kicks off the running of my app
    // can be OF_WINDOW or OF_FULLSCREEN
    // pass in width and height too:
    ofRunApp(new ofApp());

}
