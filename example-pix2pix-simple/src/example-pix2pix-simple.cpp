/*
 Very simple barebones case of pix2pix Image-to-Image translation.

 The purpose of this example is the show the most barebones way of using the msa::tf::SimpleModel API

 For a more advanced example of pix2pix see example-pix2pix (drawing with different colors and thicknesses)

 or example-pix2pix-webcam (applied to live webcam)
 */

#include "ofMain.h"
#include "ofxMSATensorFlow.h"


//--------------------------------------------------------------
//--------------------------------------------------------------
class ofApp : public ofBaseApp {
public:
    // a simple wrapper for a simple predictor model with variable number of inputs and outputs
    msa::tf::SimpleModel model;

    // a bunch of properties of the models
    // ideally should read from disk and vary with the model
    // but trying to keep the code minimal so hardcoding them since they're the same for all models
    const int input_shape[2] = {256, 256}; // dimensions {height, width} for input image
    const int output_shape[2] = {256, 256}; // dimensions {height, width} for output image
    const ofVec2f input_range = {-1, 1}; // range of values {min, max} that model expects for input
    const ofVec2f output_range = {-1, 1}; // range of values {min, max} that model outputs
    const string input_op_name = "generator/generator_inputs"; // name of op to feed input to
    const string output_op_name = "generator/generator_outputs"; // name of op to fetch output from

    // images in and out of model
    ofFloatImage img_in; // input to the model
    ofFloatImage img_out; // output from the model

    // model file management
    ofDirectory models_dir;    // data/models folder which contains subfolders for each model
    int cur_model_index = 0; // which model (i.e. folder) we're currently using


    //--------------------------------------------------------------
    void setup() {
        ofSetColor(255);
        ofBackground(0);
        ofSetVerticalSync(true);
        ofSetLogLevel(OF_LOG_VERBOSE);
        ofSetFrameRate(60);

        // scan models dir
        models_dir.listDir("models");
        if(models_dir.size()==0) {
            ofLogError() << "Couldn't find models folder." << msa::tf::missing_data_error();
            assert(false);
            ofExit(1);
        }
        models_dir.sort();
        load_model_index(0); // load first model
    }


    //--------------------------------------------------------------
    // Load graph (model trained in and exported from python) by folder NAME, and initialise session
    void load_model(string model_dir) {
        ofLogVerbose() << "loading model " << model_dir;

        // init the model
        // note that it expects arrays for input op names and output op names, so just use {}
        model.setup(ofFilePath::join(model_dir, "graph_frz.pb"), {input_op_name}, {output_op_name});
        if(! model.is_loaded()) {
            ofLogError() << "Model init error." << msa::tf::missing_data_error();
            assert(false);
            ofExit(1);
        }

        // init tensor for input. shape should be: {batch size, image height, image width, number of channels}
        // (ideally the SimpleModel graph loader would read this info from the graph_def and call this internally)
        model.init_inputs(tensorflow::DT_FLOAT, {1, input_shape[0], input_shape[1], 3});

        // load test image
        ofLogVerbose() << "loading test image";
        img_in.load(ofFilePath::join(model_dir, "test_image.png"));

        // allocating output image with correct dimensions and no alpha channel
        ofLogVerbose() << "allocating output image " << output_shape;
        img_out.allocate(output_shape[1], output_shape[0], OF_IMAGE_COLOR);
    }


    //--------------------------------------------------------------
    // Load model by folder INDEX
    void load_model_index(int index) {
        cur_model_index = ofClamp(index, 0, models_dir.size()-1);
        load_model(models_dir.getPath(cur_model_index));
    }


    //--------------------------------------------------------------
    void draw() {
        // run model on it
        if(model.is_loaded())
            model.run_image_to_image(img_in, img_out, input_range, output_range);

        // DISPLAY STUFF
        ofSetColor(255);
        int x = 0;
        img_in.draw(x, 0);
        x += img_in.getWidth();

        img_out.draw(x, 0);
        x += img_out.getWidth();

        stringstream str;
        str << ofGetFrameRate() << endl;
        str << endl;
        str << "Press number key to load model: " << endl;
        for(int i=0; i<models_dir.size(); i++) {
            auto marker = (i==cur_model_index) ? ">" : " ";
            str << " " << (i+1) << " : " << marker << " " << models_dir.getName(i) << endl;
        }

        ofDrawBitmapString(str.str(), x+20, 20);
    }

    //--------------------------------------------------------------
    void keyPressed(int key) {
        switch(key) {
        case '1':
        case '2':
        case '3':
        case '4':
        case '5':
        case '6':
        case '7':
        case '8':
        case '9':
            load_model_index(key-'1');
            break;
        }
    }


    //--------------------------------------------------------------
    void dragEvent(ofDragInfo dragInfo) {
        if(dragInfo.files.empty()) return;

        string file_path = dragInfo.files[0];

        // only PNGs work for some reason when Tensorflow is linked in
        img_in.load(file_path);
    }


};

//========================================================================
int main() {
    ofSetupOpenGL(800, 450, OF_WINDOW);
    ofRunApp(new ofApp());
}
