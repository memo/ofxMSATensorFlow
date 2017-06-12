/*
This is an example of runing a set of style transfer models

The models are trained with the python implementation in
https://github.com/olekristensen/fast-style-transfer

which is a fork that adds c++ compatible graph export to

the original fast-style-transfer by Logan Engstrom
https://github.com/lengstrom/fast-style-transfer


This example was provided by Software Artist Ole Kristensen (ole@kristensen.name)
http://ole.kristensen.name

Models are from a master project by Lilia Amundsen (liam@itu.dk) at the IT University of Copenhagen

The models were trained with the generous support from DeIC National HPC Centre, SDU on the Abacus 2.0 supercomputer (https://abacus.deic.dk)

*/

#include "ofMain.h"
#include "ofxMSATensorFlow.h"


const int img_width = 960; //1280*3/4;
const int img_height = 540; //720*3/4;


//--------------------------------------------------------------
//--------------------------------------------------------------
class ofApp : public ofBaseApp {
public:
    // a simple wrapper for a simple predictor model with variable number of inputs and outputs
    msa::tf::SimpleModel model;

    // a bunch of properties of the models
    // ideally should read from disk and vary with the model
    // but trying to keep the code minimal so hardcoding them since they're the same for all models
    const int input_shape[2] = {img_height, img_width}; // dimensions {height, width} for input image
    const int output_shape[2] = {img_height, img_width}; // dimensions {height, width} for output image
    const ofVec2f input_range = {0, 255}; // range of values {min, max} that model expects for input
    const ofVec2f output_range = {0, 255}; // range of values {min, max} that model outputs
    const string input_op_name = "img_placeholder"; // name of op to feed input to
    const string output_op_name = "add_37"; // name of op to fetch output from

    // for video capture stuff
    ofVideoGrabber video_grabber;   // for capturing from camera
    ofImage img_capture;       // captured image cropped to (height, height)
    ofImage img_processed;       // final processed image before writing to fbo
    bool capture_enabled = true;
    bool capture_flip_h = false;
    bool capture_draw = true;

    // images in and out of model
    // preallocating these to save allocating them every frame
    ofFloatImage img_in; // input to the model (read from fbo)
    ofFloatImage img_out; // output from the model

    // model file management
    ofDirectory models_dir;    // data/models folder which contains subfolders for each model
    int cur_model_index = 0; // which model (i.e. folder) we're currently using

    // other vars
    bool do_auto_run = true;    // auto run every frame

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

        // init video grabber
        video_grabber.setDeviceID(0);
        video_grabber.setUseTexture(false);
        video_grabber.setup(img_width, img_height);
    }


    //--------------------------------------------------------------
    // Load graph (model trained in and exported from python) by folder NAME, and initialise session
    void load_model(string model_dir) {
        ofLogVerbose() << "loading model " << model_dir;

        tensorflow::ConfigProto soft_config;
        soft_config.set_allow_soft_placement(true);
        soft_config.mutable_gpu_options()->set_allow_growth(true);
        tensorflow::SessionOptions session_opts;
        session_opts.config = soft_config;

        // init the model
        // note that it expects arrays for input op names and output op names, so just use {}
        model.setup(ofFilePath::join(model_dir, "of.pb"), {input_op_name}, {output_op_name}, "", "/gpu:0", nullptr, session_opts );
        if(! model.is_loaded()) {
            ofLogError() << "Model init error.";
            ofLogError() << msa::tf::missing_data_error();
            assert(false);
            ofExit(1);
        }

        // hack because of old models not properly frozen...

        // will store names of constant units
        std::vector<string> names;

        vector<tensorflow::Tensor> output_hack_tensors;      // stores all output tensors

        int node_count = model.get_graph_def()->node_size();
        ofLogNotice() << "Classifier::hack_variables - " << node_count << " nodes in graph";

        // iterate all nodes
        for(int i=0; i<node_count; i++) {
            auto n = model.get_graph_def()->node(i);
            ofLogNotice() << i << ":" << n.name(); // << n.DebugString();

            // if name contains var_hack, add to vector
            if(n.name().find("_VARHACK") != std::string::npos) {
                names.push_back(n.name());
                ofLogNotice() << "......bang";
            }
        }

        // run the network inputting the names of the constant variables we want to run
        if(!model.get_session()->Run({}, names, {}, &output_hack_tensors).ok()) {
            ofLogError() << "Error running network for weights and biases variable hack";
            assert(false);
            ofExit(1);
        }

        // init tensor for input. shape should be: {batch size, image height, image width, number of channels}
        // (ideally the SimpleModel graph loader would read this info from the graph_def and call this internally)
        model.init_inputs(tensorflow::DT_FLOAT, {1, input_shape[0], input_shape[1], 3});


        // allocate images with correct dimensions, and no alpha channel
        ofLogVerbose() << "allocating images " << input_shape;
        img_in.allocate(input_shape[1], input_shape[0], OF_IMAGE_COLOR);
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
        if(capture_enabled && video_grabber.isInitialized()) {
            // update video grabber (i.e. get next frame)
            video_grabber.update();
            if(video_grabber.isFrameNew()) {

                // crop center square and write to img_capture
                img_capture.setFromPixels(video_grabber.getPixels());
                img_capture.update();

                // flip horizontally
                img_capture.mirror(false, capture_flip_h);

                img_capture.update(); // update opengl texture with new pixels

            }
        }

        // convert to float
        ofFloatPixels fpix = img_capture.getPixels();

        // set number of channels
        fpix.setNumChannels(3);

        img_in.setFromPixels(fpix);
        img_in.update(); // update so we can draw if need be (useful for debuging)

        // run model on it
        if(do_auto_run)
            model.run_image_to_image(img_in, img_out, input_range, output_range);

        // DISPLAY STUFF
        stringstream str;
        str << ofGetFrameRate() << endl;
        str << endl;
        str << "ENTER : toggle auto run " << (do_auto_run ? "(X)" : "( )") << endl;
        str << "h     : toggle camera mirroring " << (capture_flip_h ? "(X)" : "( )") << endl;
        str << "v     : toggle camera capture " << (capture_enabled ? "(X)" : "( )") << endl;
        str << "d     : show camera image " << (capture_draw ? "(X)" : "( )") << endl;
        str << endl;

        str << "Press number key to load model: " << endl;
        for(int i=0; i<models_dir.size(); i++) {
            auto marker = (i==cur_model_index) ? ">" : " ";
            str << " " << (i+1) << " : " << marker << " " << models_dir.getName(i) << endl;
        }
        ofSetColor(255);

        img_out.draw(0,0, ofGetWidth(), ofGetHeight());

        if(capture_draw){
            // draw video input at bottom right at quarter size
            img_capture.draw(ofGetWidth()-img_capture.getWidth()/4, ofGetHeight()-img_capture.getHeight()/4, img_capture.getWidth()/4, img_capture.getHeight()/4);
        }

        // draw texts
        ofSetColor(255);
        ofDrawBitmapString(str.str(), 0, 20);

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

        case 'v':
        case 'V':
            capture_enabled ^= true;
            break;

        case 'h':
        case 'H':
            capture_flip_h ^= true;
            break;

        case 'd':
        case 'D':
            capture_draw ^= true;
            break;

        case OF_KEY_RETURN:
            do_auto_run ^= true;
            break;
        }
    }



    //--------------------------------------------------------------
    void mouseDragged( int x, int y, int button) {
    }



    //--------------------------------------------------------------
    void mousePressed( int x, int y, int button) {
    }



    //--------------------------------------------------------------
    virtual void mouseReleased(int x, int y, int button) {
    }



    //--------------------------------------------------------------
    void dragEvent(ofDragInfo dragInfo) {
    }


};

//========================================================================
int main() {
    ofSetupOpenGL(img_width, img_height, OF_WINDOW);
    ofRunApp(new ofApp());
}

// ///////////////////////////////////////////////////////////////////////////////
