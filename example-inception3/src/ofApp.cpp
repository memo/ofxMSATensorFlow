/*
 * Image recognition using Google's Inception network
 * based on https://www.tensorflow.org/versions/master/tutorials/image_recognition/index.html
 *
 *
 * Uses pre-trained model https://storage.googleapis.com/download.tensorflow.org/models/inception_dec_2015.zip
 *
 * openFrameworks code loads and processes pre-trained model (i.e. makes calculations/predictions)
 *
 */

#include "ofApp.h"

// input image dimensions dictated by trained model
#define kInputWidth     299
#define kInputHeight    299
#define kInputSize      (kInputWidth * kInputHeight)


// we need to normalize the images before feeding into the network
// from each pixel we subtract the mean and divide by variance
// this is also dictated by the trained model
#define kInputMean      (128.0f/255.0f)
#define kInputStd       (128.0f/255.0f)

// model & labels files to load
#define kModelPath      "models/tensorflow_inception_graph.pb"
#define kLabelsPath     "models/imagenet_comp_graph_label_strings.txt"


// every node in the network has a name
// when passing in data to the network, or reading data back, we need to refer to the node by name
// i.e. 'pass this data to node A', or 'read data back from node X'
// these node names are specific to the architecture of the model
#define kInputLayer     "Mul"
#define kOutputLayer    "softmax"



//--------------------------------------------------------------
// ofImage::load() (ie. Freeimage load) doesn't work with TensorFlow! (See README.md)
// so I have to resort to this awful trick of loading raw image data 299x299 RGB
void loadImageRaw(string path, ofImage &img) {
    ofFile file(path);
    img.setFromPixels((unsigned char*)file.readToBuffer().getData(), kInputWidth, kInputHeight, OF_IMAGE_COLOR);
}



//--------------------------------------------------------------
// Takes a file name, and loads a list of labels from it, one per line, and
// returns a vector of the strings. It pads with empty strings so the length
// of the result is a multiple of 16, because our model expects that.
bool ReadLabelsFile(string file_name, std::vector<string>* result) {
    std::ifstream file(file_name);
    if (!file) {
        ofLogError() <<"ReadLabelsFile: " << file_name << " not found.";
        return false;
    }

    result->clear();
    string line;
    while (std::getline(file, line)) {
        result->push_back(line);
    }
    const int padding = 16;
    while (result->size() % padding) {
        result->emplace_back();
    }
    return true;
}


//---------------------------------------------------------
// Load pixels into the network, get the results
void ofApp::classify(ofPixels &pix) {

    // interesting workaround I need to do to convert unsiged char pix to float
    ofImage temp;
    temp.setFromPixels(pix);    // this is an unsigned char image with the same pixels
    processed_image = temp;     // convert unsigned char image to float image

    // need to resize image to specific dimensions the model is expecting
    processed_image.resize(kInputWidth, kInputHeight);

    // pixelwise normalize image by subtracting the mean and dividing by variance (across entire dataset)
    // I could do this without iterating over the pixels, by setting up a TensorFlow Graph, but I can't be bothered, this is less code
    float* pix_data = processed_image.getPixels().getData();
    if(!pix_data) {
        ofLogError() << "Could not classify. pixel data is NULL";
        return;
    }
    for(int i=0; i<kInputSize*3; i++) pix_data[i] = pix_data[i] = (pix_data[i] - kInputMean) / kInputStd;

    //  make sure opengl texture is updated with new pixel info (needed for correct rendering)
    processed_image.update();

    // copy data from image into tensorflow's Tensor class
    ofxMSATensorFlow::imageToTensor(processed_image, image_tensor);

    // feed the data into the network, and request output
    // output_tensors don't need to be initialized or allocated. they will be filled once the network runs
    if( !msa_tf.run({ {kInputLayer, image_tensor } }, { kOutputLayer }, {}, &output_tensors) ) {
        ofLogError() << "Error during running. Check console for details." << endl;
        return;
    }

    // the output from the network above is an array of probabilities for every single label
    // i.e. thousands of probabilities, we only want to the top few
    ofxMSATensorFlow::getTopScores(output_tensors[0], 6, top_label_indices, top_scores);
}



//--------------------------------------------------------------
void ofApp::loadNextImage() {
    static int file_index = 0;

    // System load dialog doesn't work with tensorflow :(
    //auto o = ofSystemLoadDialog("Select image");
    //if(!o.bSuccess) return;

    // FreeImage doesn't work with tensorflow! :(
    //img.load("images/fanboy.jpg");

    // resorting to awful raw data file load hack!
    loadImageRaw(image_dir.getPath(file_index), input_image);
    classify(input_image.getPixels());
    file_index = (file_index+1) % image_dir.getFiles().size();
}


//--------------------------------------------------------------
void ofApp::setup(){
    ofLogNotice() << "Initializing... ";
    ofBackground(0);
    ofSetVerticalSync(true);
    ofSetFrameRate(60);

    // get a list of all images in the 'images' folder
    image_dir.listDir("images");

    // Initialize tensorflow session, return if error
    if( !msa_tf.setup() ) return;

    // Load graph (i.e. trained model) add to session, return if error
    if( !msa_tf.loadGraph(kModelPath) ) return;

    // load text file containing labels (i.e. associating classification index with human readable text)
    if( !ReadLabelsFile(ofToDataPath(kLabelsPath), &labels) ) return;

    // initialize input tensor dimensions
    // (not sure what the best way to do this was as there isn't an 'init' method, just a constructor)
    image_tensor = tensorflow::Tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({ 1, kInputHeight, kInputWidth, 3 }));

    // load first image to classify
    loadNextImage();

    ofLogNotice() << "Init successfull";
}


//--------------------------------------------------------------
void ofApp::update(){

    // if video_grabber active,
    if(video_grabber) {
        // grab frame
        video_grabber->update();

        if(video_grabber->isFrameNew()) {

            // update input_image so it's drawn in the right place
            input_image.setFromPixels(video_grabber->getPixels());

            // send to classification if keypressed
            if(ofGetKeyPressed(' ')) classify(input_image.getPixels());
        }
    }
}

//--------------------------------------------------------------
void ofApp::draw(){
    // draw input image if it's available
    float x = 0;
    if(input_image.isAllocated()) {
        input_image.draw(x, 0);
        x += input_image.getWidth();
    }

    // draw processed image if it's available
    //    if(processed_image.isAllocated()) {
    //        processed_image.draw(x, 0);
    //        x += input_image.getWidth();
    //    }

    x += 20;
    float w = ofGetWidth() - 400 - x;
    float y = 40;
    float bar_height = 35;

    // iterate top scores and draw them
    for(int i=0; i<top_scores.size(); i++) {
        int label_index = top_label_indices[i];
        string label = labels[label_index];
        float p = top_scores[i];    // the score (i.e. probability, 0...1)

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


    ofSetColor(255);
    ofDrawBitmapString(ofToString(ofGetFrameRate()), ofGetWidth() - 100, 30);

    stringstream str_inst;
    str_inst << "'l' to load image\n";
    str_inst << "or drag an image (must be raw, 299x299) onto the window\n";
    str_inst << "'v' to toggle video input";
    ofDrawBitmapString(str_inst.str(), 15, input_image.getHeight() + 30);
}


//--------------------------------------------------------------
void ofApp::keyPressed(int key){
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
void ofApp::dragEvent(ofDragInfo dragInfo){
    if(dragInfo.files.empty()) return;

    string filePath = dragInfo.files[0];
    //img.load(filePath);  // FreeImage doesn't work :(
    loadImageRaw(filePath, input_image);
    classify(input_image.getPixels());
}
