/*
 * Image recognition using Google's Inception v3 network
 * based on https://www.tensorflow.org/versions/master/tutorials/image_recognition/index.html
 *
 * Uses pre-trained model https://storage.googleapis.com/download.tensorflow.org/models/inception_dec_2015.zip
 *
 * openFrameworks code loads and processes pre-trained model (i.e. makes calculations/predictions)
 *
 */



#include "ofMain.h"
#include "ofxMSATensorFlow.h"


//--------------------------------------------------------------
// ofImage::load() (ie. Freeimage load) doesn't work with TensorFlow! (See README.md)
// so I have to resort to this awful trick of loading raw image data 299x299 RGB
static void loadImageRaw(string path, ofImage &img, int w, int h) {
    ofFile file(path);
    img.setFromPixels((unsigned char*)file.readToBuffer().getData(), w, h, OF_IMAGE_COLOR);
}



class ofApp : public ofBaseApp {
public:

    // classifies pixels
    // check the src of this class (ofxMSATFImageClassifier) to see how to do more generic stuff with ofxMSATensorFlow
    msa::tf::ImageClassifier classifier;

    // for webcam input
    shared_ptr<ofVideoGrabber> video_grabber;

    // folder of images to classify
    ofDirectory image_dir;

    // top scoring classes
    vector<int> top_label_indices;  // contains top n label indices for input image
    vector<float> top_class_probs;  // contains top n probabilities for current input image


    //--------------------------------------------------------------
    void loadNextImage() {
        static int file_index = 0;
        ofImage img;

        // System load dialog doesn't work with tensorflow :(
        //auto o = ofSystemLoadDialog("Select image");
        //if(!o.bSuccess) return;

        // FreeImage doesn't work with tensorflow! :(
        //img.load("images/fanboy.jpg");

        // resorting to awful raw data file load hack!
        loadImageRaw(image_dir.getPath(file_index), img, 299, 299);

        classify(img.getPixels());
        file_index = (file_index+1) % image_dir.getFiles().size();
    }


    //--------------------------------------------------------------
    void classify(const ofPixels& pix) {
        // classify pixels
        classifier.classify(pix);

        msa::tf::getTopScores(classifier.getOutputTensors()[0], 6, top_label_indices, top_class_probs);
    }

    //--------------------------------------------------------------
    void setup(){
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

        // get a list of all images in the 'images' folder
        image_dir.listDir("images");

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

            // if video grabber active, draw in bottom left corner
            if(video_grabber) video_grabber->draw(0, ofGetHeight() - 240, 320, 240);


            float x = 0;

            // draw input image
            classifier.getInputImage().draw(x, 0);
            x += classifier.getInputImage().getWidth();

            // draw processed image
            classifier.getProcessedImage().draw(x, 0);
            x += classifier.getProcessedImage().getWidth();

            x += 20;

            float w = ofGetWidth() - 400 - x;
            float y = 40;
            float bar_height = 35;


            // iterate top scores and draw them
            for(int i=0; i<top_class_probs.size(); i++) {
                int label_index = top_label_indices[i];
                string label = classifier.getLabels()[label_index];
                float p = top_class_probs[i];    // the score (i.e. probability, 0...1)

                // draw full bar
                ofSetColor(ofLerp(50.0, 255.0, p), ofLerp(100.0, 0.0, p), ofLerp(150.0, 0.0, p));
                ofDrawRectangle(x, y, w * p, bar_height);
                ofSetColor(40);

                // draw outline
                ofNoFill();
                ofDrawRectangle(x, y, w, bar_height);
                ofFill();

                // draw text
                ofSetColor(255);
                ofDrawBitmapString(label + " (" + ofToString(label_index) + "): " + ofToString(p,4), x + w + 10, y + 20);
                y += bar_height + 5;
            }
        }

        ofSetColor(255);
        ofDrawBitmapString(ofToString(ofGetFrameRate()), ofGetWidth() - 100, 30);

        stringstream str_inst;
        str_inst << "'l' to load image\n";
        str_inst << "or drag an image (must be raw, 299x299) onto the window\n";
        str_inst << "'v' to toggle video input";
        ofDrawBitmapString(str_inst.str(), 15, classifier.getHeight() + 30);
    }


    //--------------------------------------------------------------
    void keyPressed(int key){
        switch(key) {

        case 'v':
            if(video_grabber) video_grabber = NULL;
            else {
                video_grabber = make_shared<ofVideoGrabber>();
                video_grabber->setup(320, 240);
            }
            break;

        case 'l':
            loadNextImage();
            break;
        }
    }

    //--------------------------------------------------------------
    void dragEvent(ofDragInfo dragInfo){
        if(dragInfo.files.empty()) return;

        ofImage img;

        string filePath = dragInfo.files[0];
        //img.load(filePath);  // FreeImage doesn't work :(
        loadImageRaw(filePath, img, 299, 299);
        classify(img.getPixels());
    }

};



//========================================================================
int main( ){
    ofSetupOpenGL(1200, 800, OF_WINDOW);			// <-------- setup the GL context

    // this kicks off the running of my app
    // can be OF_WINDOW or OF_FULLSCREEN
    // pass in width and height too:
    ofRunApp(new ofApp());

}
