/*
 * Image recognition using Google's Inception v3 network
 * based on https://www.tensorflow.org/tutorials/image_recognition#usage_with_the_c_api
 *
 * Uses pre-trained model https://storage.googleapis.com/download.tensorflow.org/models/inception_dec_2015.zip
 *
 * openFrameworks code loads and processes pre-trained model (i.e. makes calculations/predictions)
 *
 */



#include "ofMain.h"
#include "ofxMSATensorFlow.h"


class ofApp : public ofBaseApp {
public:

    // classifies pixels
    // check the src of this class (ofxMSATFImageClassifier) to see how to do more generic stuff with ofxMSATensorFlow
    // UPDATE: Actually the msa::tf::SimpleModel supercedes this. Need to update it.
    msa::tf::ImageClassifier classifier;

    // for webcam input
    shared_ptr<ofVideoGrabber> video_grabber;

    // contents of folder with images to classify
    vector<ofFile> image_files;

    // current image being classified
    ofImage img;

    // top scoring classes
    vector<float> top_class_probs;  // contains top k probabilities for current image
    vector<int> top_label_indices;  // contains top k label indices for current image


    //--------------------------------------------------------------
    void loadNextImage() {
        static int file_index = 0;
        // System load dialog doesn't work with tensorflow (at least on ubuntu, see FAQ):(
//        auto o = ofSystemLoadDialog("Select image");
//        if(!o.bSuccess) return;

        // only PNGs work when tensorflow is linked in (at least on ubuntu), see FAQ :(
        img.load(image_files[file_index].getAbsolutePath());
        if(img.isAllocated()) classify(img.getPixels());

        file_index = (file_index+1) % image_files.size();
    }


    //--------------------------------------------------------------
    void classify(const ofPixels& pix) {
        // classify pixels
        classifier.classify(pix);

        msa::tf::get_topk(classifier.getClassProbs(), top_label_indices, top_class_probs, 10);
    }

    //--------------------------------------------------------------
    void setup() {
        ofLogNotice() << "Initializing... ";
        ofBackground(0);
        ofSetVerticalSync(true);
        //        ofSetFrameRate(60);

        // initialize the image classifier, lots of params to setup
        // these settings are specific to the model
        msa::tf::ImageClassifier::Settings settings;
        settings.image_dims = { 299, 299, 3 };
        settings.itensor_dims = { 1, 299, 299, 3 };
        settings.model_path = "models/tensorflow_inception_graph.pb";
        settings.labels_path = "models/imagenet_comp_graph_label_strings.txt";
        settings.input_layer_name = "Mul";
        settings.output_layer_name = "softmax";
        settings.dropout_layer_name = "";
        settings.varconst_layer_suffix = "_VARHACK";
        settings.norm_mean = 128.0f/255.0f;
        settings.norm_stddev = 128.0f/255.0f;

        // initialize classifier with these settings
        classifier.setup(settings);
        if(!classifier.getGraphDef()) {
            ofLogError() << "Could not initialize session. Did you download the data files and place them in the data folder? ";
            ofLogError() << "Download from https://github.com/memo/ofxMSATensorFlow/releases";
            ofLogError() << "More info at https://github.com/memo/ofxMSATensorFlow/wiki";
            assert(false);
            ofExit(1);
        }

        // get a list of all images in the 'images' folder
        ofDirectory image_dir;
        image_dir.listDir("images");
        image_files = image_dir.getFiles();

        // load first image to classify
        loadNextImage();

        ofLogNotice() << "Init successfull";
    }


    //--------------------------------------------------------------
    void update() {
        // if video_grabber active,
        if(video_grabber) {
            // grab frame
            video_grabber->update();

            if(video_grabber->isFrameNew()) {
                // send to classification if keypressed
                if(ofGetKeyPressed(' '))
                    classify(video_grabber->getPixels());
            }
        }
    }


    //--------------------------------------------------------------
    void draw() {
        if(classifier.isReady()) {
            ofSetColor(255);

            float x = 0;

            // draw input image
            classifier.getInputImage().draw(x, 0);
            x += classifier.getInputImage().getWidth();

            // draw processed image
            classifier.getProcessedImage().draw(x, 0);
            x += classifier.getProcessedImage().getWidth();

            x += 20;

            float w = ofGetWidth() - x;
            float y = 40;
            float bar_height = 35;


            // iterate top scores and draw them
            for(int i=0; i<top_class_probs.size(); i++) {
                int label_index = top_label_indices[i];
                string label = classifier.getLabels()[label_index];
                float p = top_class_probs[i];    // the score (i.e. probability, 0...1)

                // draw full bar
                ofSetColor(ofLerp(0.0, 255.0, p), 0, ofLerp(255.0, 0.0, p));
                ofDrawRectangle(x, y, w * p, bar_height);
                ofSetColor(40);

                // draw outline
                ofNoFill();
                ofDrawRectangle(x, y, w, bar_height);
                ofFill();

                // draw text
                ofSetColor(255);
                ofDrawBitmapString(label + " (" + ofToString(label_index) + "): " + ofToString(p,4), x + 10, y + 20);
                y += bar_height + 5;
            }

            classifier.draw_probs(ofRectangle(0, ofGetHeight()/2, ofGetWidth(), ofGetHeight()/2));
        }

        stringstream str_inst;
        str_inst << "'l' to load image\n";
        str_inst << "or drag an image (must be PNG) onto the window\n";
        str_inst << "\n";
        str_inst << "'v' to toggle video input\n";


        // draw video grabber if active
        if(video_grabber) {
            str_inst << "Press SPACE to classify\n";
            ofSetColor(255);
            video_grabber->draw(0, 0, 320, 240);
        }

        ofSetColor(255);
        ofDrawBitmapString(ofToString(ofGetFrameRate()), ofGetWidth() - 100, 30);
        ofDrawBitmapString(str_inst.str(), 15, classifier.getHeight() + 30);
    }


    //--------------------------------------------------------------
    void keyPressed(int key) {
        switch(key) {

        case 'v':
            if(video_grabber) video_grabber = NULL;
            else {
                // init video grabber
                video_grabber = make_shared<ofVideoGrabber>();
                auto devices = video_grabber->listDevices();
                if(devices.size() > 0) {
                    video_grabber->setDeviceID(devices.back().id);
                    video_grabber->setup(640, 480);
                } else {
                    video_grabber = nullptr;
                }
            }
            break;

        case 'l':
            loadNextImage();
            break;
        }
    }

    //--------------------------------------------------------------
    void dragEvent(ofDragInfo dragInfo) {
        if(dragInfo.files.empty()) return;

        string file_path = dragInfo.files[0];

        // only PNGs work for some reason when Tensorflow is linked in
        ofImage img;
        img.load(file_path);
        if(img.isAllocated()) classify(img.getPixels());
    }

};



//========================================================================
int main() {
    ofSetupOpenGL(1200, 800, OF_WINDOW);
    ofRunApp(new ofApp());
}
