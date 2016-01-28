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
#include "ImageHelpers.h"


// input image dimensions dictated by trained model
#define kInputWidth     299
#define kInputHeight    299
#define kInputSize      (kInputWidth * kInputHeight)

#define kInputMean      128
#define kInputStd       128


#define kTestImagePath  "images/grace_hopper.jpg"


// model & labels files to load
#define kModelPath      "models/tensorflow_inception_graph.pb"
#define kLabelsPath     "models/imagenet_comp_graph_label_strings.txt"

// layer names
#define kInputLayer     "Mul"
#define kOutputLayer    "softmax"



void loadImageRaw(string path, ofImage &img) {
    ofFile file(path);
    img.setFromPixels((unsigned char*)file.readToBuffer().getData(), kInputWidth, kInputHeight, OF_IMAGE_COLOR);
}

//--------------------------------------------------------------
void ofApp::setup(){
    ofLogNotice() << "Initializing... ";
    ofSetColor(255);
    ofBackground(0);
    ofSetVerticalSync(true);

    mouse_painter.setup(299);

    // Initialize tensorflow session, return if error
    if( !msa_tf.setup() ) return;

    // Load graph (i.e. trained model) add to session, return if error
    if( !msa_tf.loadGraph(kModelPath) ) return;

    // initialize input tensor dimensions
    // (not sure what the best way to do this was as there isn't an 'init' method, just a constructor)
    x_inputs = tensorflow::Tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({ 1, kInputHeight, kInputWidth, 3 }));

    if( !ReadLabelsFile(ofToDataPath(kLabelsPath), &labels) ) return;

    loadNextImage();

    ofLogNotice() << "Init successfull";
}

//---------------------------------------------------------
void ofApp::runModel() {
    // Run the graph, pass in our inputs and desired outputs, evaluate operation and return
    if( !msa_tf.run({ {kInputLayer, x_inputs } }, { kOutputLayer }, {}, &outputs) )
        ofLogError() << "Error during running. Check console for details." << endl;
}

//--------------------------------------------------------------
void ofApp::update(){
    // get pixels from mousepainter and resize to correct dimensions
    resized_img.setFromPixels(mouse_painter.get_pixels());
    resized_img.resize(kInputWidth, kInputHeight);
    resized_img.setImageType(OF_IMAGE_COLOR);

    // convert to float and normalize
    // there's probably a fancier way to copy these values over and normalize, but this works
    processed_img = resized_img;
    float* fimg_data = processed_img.getPixels().getData();

    float mean = kInputMean / 256.0f;
    float std = kInputStd / 256.0f;

    auto x_flat_data = x_inputs.flat<float>().data();    // get tensorflow::Tensor data as a flattened Eigen::Tensor
    for(int i=0; i<kInputSize*3; i++) {
        x_flat_data[i] =
                fimg_data[i] = (fimg_data[i] - mean) / std;
    }
    processed_img.update();


    if(msa_tf.isReady() && (ofGetFrameNum() % 60 == 0)) runModel();
}

//--------------------------------------------------------------
void ofApp::draw(){

    // if video_grabber active, draw into active square
    if(video_grabber) {
        video_grabber->update();
        mouse_painter.drawIntoMe(*video_grabber);
    }

    // draw mouse painter
    mouse_painter.draw();

    //    resized_img.draw(mouse_painter.getWidth()*2, 0);
    processed_img.draw(mouse_painter.getWidth(), 0);

    if(msa_tf.isReady() && !outputs.empty() && outputs[0].IsInitialized()) {

        const int how_many_labels = 10;
        Tensor indices;
        Tensor scores;
        GetTopLabels(outputs, how_many_labels, &indices, &scores);
        tensorflow::TTypes<float>::Flat scores_flat = scores.flat<float>();
        tensorflow::TTypes<int32>::Flat indices_flat = indices.flat<int32>();

        float x = mouse_painter.getWidth() * 2 + 10;
        float w = ofGetWidth() - 300 - x;
        float y = 30;
        float bar_height = 50;


        for(int i=0; i<how_many_labels; i++) {
            const int label_index = indices_flat(i);
            const float p = scores_flat(i);

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
            ofDrawBitmapString(labels.at(label_index) + " (" + ofToString(label_index) + "): " + ofToString(p,4), x + w, y + 30);
            y += bar_height + 5;
        }
    }

    ofSetColor(255);
    ofDrawBitmapString(ofToString(ofGetFrameRate()), ofGetWidth() - 100, 30);

    stringstream str_inst;
    str_inst << "Paint in the box above\n";
    str_inst << "Right-click to erase\n";
    str_inst << "'c' to clear\n";
    str_inst << endl;
    str_inst << "'l' to load image\n";
    str_inst << "'v' to toggle video input";
    ofDrawBitmapString(str_inst.str(), 15, mouse_painter.getHeight() + 15);

    if(video_grabber) video_grabber->draw(0, 0);
}

static string files[] = {"images/grace_hopper299.data, images/fanboy299.data"};
static int file_index = 0;
void ofApp::loadNextImage() {
    //auto o = ofSystemLoadDialog("Select image"); // doesn't work with tensorflow :(
    //if(!o.bSuccess) return;
    ofImage img;
    //img.load("images/fanboy.jpg");    // FreeImage doesn't work with tensorflow! :(
    //loadImageRaw(o.getPath(), img);
    loadImageRaw(files[file_index], img);
    mouse_painter.drawIntoMe(img);
    file_index = (file_index+1) % 2;
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){
    switch(key) {
    case 'c':
        mouse_painter.clear();
        break;

    case 'v':
        if(video_grabber) video_grabber = NULL;
        else {
            video_grabber = make_shared<ofVideoGrabber>();
            video_grabber->setup(320, 240);
        }
        break;

    case 'l':
        loadNextImage();
        runModel();
        break;
    }
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){
    mouse_painter.penDrag(ofVec2f(x, y), button==2, 150);
}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){
    mouse_painter.penDown(ofVec2f(x, y), button==2, 150);
}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){
    mouse_painter.penUp();
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

//---------------------------tenso-----------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){
    if(dragInfo.files.empty()) return;

    string filePath = dragInfo.files[0];
    ofImage img;
    //img.load(filePath);  // FreeImage doesn't work :(
    loadImageRaw(filePath, img);
    mouse_painter.drawIntoMe(img);
    runModel();
}
