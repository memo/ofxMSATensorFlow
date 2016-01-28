/*
 * Unit tests
 *
 */

#include "ofApp.h"

#define kInputWidth 256
#define kInputHeight 256
#define kInputChannels 3
#define kInputElements (kInputWidth * kInputHeight * kInputChannels)

#define kNumImages 11

using namespace tensorflow;

ofFloatImage img[kNumImages];

void loadImageRaw(string path, int w, int h, ofImage &img) {
    ofFile file(path);
    img.setFromPixels((unsigned char*)file.readToBuffer().getData(), w, h, OF_IMAGE_COLOR);
}


//--------------------------------------------------------------
// check all vector <--> tensor <--> pixel etc methods and combinations
void testCopyMethods() {

    // load image
    ofImage img_orig;
    //img.load("images/fanboy.jpg");    // FreeImage doesn't work with tensorflow! :(
    loadImageRaw(ofToDataPath("images/grace_hopper256.data"), kInputWidth, kInputHeight, img_orig);

    // allocate images
    for(int i=0; i<kNumImages; i++) img[i].allocate(kInputWidth, kInputHeight, OF_IMAGE_COLOR);

    img[0] = img_orig;

    bool do_memcpy = true;

    Tensor tensor;

    // bounce data around a bunch of different data types and size, see if data is preserved
    tensor = Tensor(DT_FLOAT, tensorflow::TensorShape({ kInputChannels, kInputWidth, kInputHeight}));
    ofxMSATensorFlow::pixelsToTensor(img[0].getPixels(), tensor, do_memcpy);

    ofxMSATensorFlow::tensorToPixels(tensor, img[1].getPixels(), do_memcpy);
    img[1].update();

    tensor = Tensor(DT_FLOAT, tensorflow::TensorShape({ kInputWidth / 2, kInputHeight * kInputChannels * 2}));
    ofxMSATensorFlow::arrayToTensor(img[1].getPixels().getData(), tensor, do_memcpy);

    vector<float> v1;
    ofxMSATensorFlow::tensorToVector(tensor, v1, do_memcpy);
    assert(v1.size() == kInputElements);

    tensor = Tensor(DT_FLOAT, tensorflow::TensorShape({ kInputElements }));
    ofxMSATensorFlow::vectorToTensor(v1, tensor, do_memcpy);

    ofxMSATensorFlow::tensorToArray(tensor, img[2].getPixels().getData(), do_memcpy);
    img[2].update();

    tensor = Tensor(DT_FLOAT, tensorflow::TensorShape({ kInputChannels*16, kInputHeight/16, kInputWidth/32, 32}));
    ofxMSATensorFlow::imageToTensor(img[2], tensor, do_memcpy);
    ofxMSATensorFlow::tensorToImage(tensor, img[3], do_memcpy);

    tensor = Tensor(DT_FLOAT, tensorflow::TensorShape({ kInputChannels, kInputWidth, kInputHeight }));
    ofxMSATensorFlow::imageToTensor(img[3], tensor, do_memcpy);

    // CLEAR ALL IMAGES TO TEST AUTO ALLOCATIONS
    for(int i=4; i<kNumImages; i++) img[i].clear();

    ofxMSATensorFlow::tensorToImage(tensor, img[4], do_memcpy);

    tensor = Tensor(DT_FLOAT, tensorflow::TensorShape({ kInputChannels, kInputWidth*2, kInputHeight/2 }));
    ofxMSATensorFlow::imageToTensor(img[4], tensor, do_memcpy);

    // it's normal that this looks mangled, it's not supposed to be resized, just reshaped
    ofxMSATensorFlow::tensorToImage(tensor, img[5], do_memcpy);

    tensor = Tensor(DT_FLOAT, tensorflow::TensorShape({ kInputChannels, kInputWidth, kInputHeight }));
    ofxMSATensorFlow::imageToTensor(img[5], tensor, do_memcpy);

    ofxMSATensorFlow::tensorToImage(tensor, img[6], do_memcpy);

    // convert to grayscale
    ofImage img_temp;
    img_temp= img[6];
    img_temp.setImageType(OF_IMAGE_GRAYSCALE);
    img[7] = img_temp;

    tensor = Tensor(DT_FLOAT, tensorflow::TensorShape({ kInputWidth, kInputHeight }));
    ofxMSATensorFlow::imageToTensor(img[7], tensor, do_memcpy);

    ofxMSATensorFlow::tensorToImage(tensor, img[8], do_memcpy);

    tensor = Tensor(DT_FLOAT, tensorflow::TensorShape({ kInputWidth/2, kInputHeight*2 }));
    ofxMSATensorFlow::imageToTensor(img[8], tensor, do_memcpy);

    ofxMSATensorFlow::tensorToImage(tensor, img[9], do_memcpy);

    tensor = Tensor(DT_FLOAT, tensorflow::TensorShape({ kInputWidth, kInputHeight }));
    ofxMSATensorFlow::imageToTensor(img[9], tensor, do_memcpy);

    ofxMSATensorFlow::tensorToImage(tensor, img[10], do_memcpy);
}

//--------------------------------------------------------------
void ofApp::setup(){
    ofSetColor(255);
    ofBackground(0);
    ofSetVerticalSync(true);

    testCopyMethods();
}



//--------------------------------------------------------------
void ofApp::update(){

}

//--------------------------------------------------------------
void ofApp::draw(){

    int x=0, y=0;
    for(int i=0; i<kNumImages; i++) {
        if(x + kInputWidth >= ofGetWidth()) {
            x = 0;
            y += kInputHeight;
        }
        img[i].draw(x, y);
        x += img[i].getWidth();
    }
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){


}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){

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

}
