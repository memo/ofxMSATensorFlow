/*
 * Two different MNIST classification examples
 * - Very simple (and not very good)  multinomial logistic regression based on https://www.tensorflow.org/versions/master/tutorials/mnist/beginners/index.html
 * - Deep (better, but more code to follow) based on https://www.tensorflow.org/versions/master/tutorials/mnist/pros/index.html
 *
 * Python script downloads training data, constructs tensor flow graph, trains and exports binary model file (see bin/py)
 *
 * openFrameworks code loads and processes pre-trained model (i.e. makes calculations/predictions)
 *
 */


#include "ofMain.h"
#include "ofxMSATensorFlow.h"
#include "MousePainter.h"


class ofApp : public ofBaseApp {
public:

    // classifies pixels
    // check the src of this class (ofxMSATFImageClassifier) to see how to do more generic stuff with ofxMSATensorFlow
    // UPDATE: Actually the msa::tf::SimpleModel supercedes this. Need to update it.
    msa::tf::ImageClassifier classifier;

    // simple visualization of weights layer,
    // only really meaningful on the single layer simple model
    // deeper networks need more complex visualization ( see http://arxiv.org/abs/1311.2901 )
    msa::tf::LayerVisualizer layer_viz;

    // class for painting with mouse, used as input image
    MousePainter mouse_painter;

    // which model to use
    bool use_deep_model;


    //--------------------------------------------------------------
    void init_classifier(bool use_deep_model) {
        this->use_deep_model = use_deep_model;

        // initialize the image classifier, lots of params to setup
        msa::tf::ImageClassifier::Settings settings;

        // these settings are specific to the model
        // settings which are common to both models
        settings.image_dims = { 28, 28, 1 };
        settings.itensor_dims = { 1, 28 * 28 };
        settings.labels_path = "";
        settings.input_layer_name = "x_inputs";
        settings.output_layer_name = "y_outputs";
        settings.dropout_layer_name = "";
        settings.varconst_layer_suffix = "_VARHACK";
        settings.norm_mean = 0;
        settings.norm_stddev = 0;

        // settings which are specific to the individual models
        if(use_deep_model) {
            settings.model_path = "model-deep/mnist-deep.pb";   // load appropiate model file
            settings.dropout_layer_name = "keep_prob";    // this model has dropout, so need to set keep probability to 1.0
        } else {
            settings.model_path = "model-simple/mnist-simple.pb";   // load appropiate model file
            settings.dropout_layer_name = "";   // this model doesn't have dropout, so ignore
        }

        // initialize classifier with these settings
        classifier.setup(settings);
        if(!classifier.getGraphDef()) {
            ofLogError() << "Session init error." << msa::tf::missing_data_error();
            assert(false);
            ofExit(1);
        }

        // initialize layer visualizer
        layer_viz.setup(classifier.getSession(), classifier.getGraphDef(), "VIZ_VARHACK");
    }


    //--------------------------------------------------------------
    void setup() {
        ofLogNotice() << "Initializing... ";
        ofBackground(50);
        ofSetVerticalSync(true);

        // setup mouse painter
        mouse_painter.setup(256);

        // initialize classifier
        init_classifier(true);

        ofLogNotice() << "Init successfull";
    }


    //--------------------------------------------------------------
    void update() {
        if(classifier.isReady()) {
            classifier.classify(mouse_painter.get_pixels());
        }
    }


    //--------------------------------------------------------------
    void draw() {
        // draw mouse painter
        mouse_painter.draw();

        stringstream str_outputs;


        if(classifier.isReady() && classifier.getNumClasses() > 0) {
            float cur_y = mouse_painter.getHeight() + 10;
            ofSetColor(255);

            // DRAW LAYER PARAMETERS (only really useful for the single layer version, deeper networks need more complex visualization)
            cur_y += layer_viz.draw(0, cur_y, ofGetWidth());


            // DRAW OUTPUT PROBABILITY BARS
            float box_spacing = ofGetWidth() / classifier.getNumClasses();
            float box_width = box_spacing * 0.8;

            for(int i=0; i<classifier.getNumClasses(); i++) {
                float p = classifier.getClassProbs()[i]; // probability of this label

                // draw probability bar
                float h = (ofGetHeight() - cur_y) * p;
                float x = ofMap(i, 0, classifier.getNumClasses()-1, 0, ofGetWidth() - box_spacing);
                x += (box_spacing - box_width)/2;

                ofSetColor(ofLerp(50.0, 255.0, p), ofLerp(100.0, 0.0, p), ofLerp(150.0, 0.0, p));
                ofDrawRectangle(x, ofGetHeight(), box_width, -h);

                str_outputs << ofToString(classifier.getClassProbs()[i], 3) << " ";

                // draw text
                ofDrawBitmapString(ofToString(i) + ": " + ofToString(p, 2), x, ofGetHeight() - h - 10);
            }


            // draw line indicating top score
            ofSetColor(200);
            ofDrawLine(0, cur_y, ofGetWidth(), cur_y);
        }


        stringstream str;
        str << "Paint in the box" << endl;
        str << "Rightclick to erase" << endl;
        str << "'c' to clear" << endl;
        str << endl;
        str << "'m' to toggle model. Currently using " << classifier.settings.model_path << endl;
        str << endl;
        str << "Outputs: " << str_outputs.str() << endl;
        str << endl;
        str << "fps: " << ofToString(ofGetFrameRate(), 2) << endl;

        ofSetColor(255);
        ofDrawBitmapString(str.str(), mouse_painter.getWidth() + 20, 30);
    }

    //--------------------------------------------------------------
    void keyPressed(int key) {
        switch(key) {
        case 'c': mouse_painter.clear(); break;
        case 'm': init_classifier( !use_deep_model); break; // toggle model between deep and simple
        }
    }

    //--------------------------------------------------------------
    void mouseDragged(int x, int y, int button) {
        mouse_painter.penDrag(ofVec2f(x, y), button==2);
    }

    //--------------------------------------------------------------
    void mousePressed(int x, int y, int button) {
        mouse_painter.penDown(ofVec2f(x, y), button==2);
    }

    //--------------------------------------------------------------
    void mouseReleased(int x, int y, int button) {
        mouse_painter.penUp();
    }
};


//========================================================================
int main() {
    ofSetupOpenGL(1600, 800, OF_WINDOW);
    ofRunApp(new ofApp());
}
