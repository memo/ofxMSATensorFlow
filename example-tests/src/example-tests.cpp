/*
 * Unit tests
 *
 */

#include "ofMain.h"
#include "ofxMSATensorFlow.h"

#define kNumIterations 11

using namespace tensorflow;


class ofApp : public ofBaseApp {
public:
    ofFloatImage img[kNumIterations];

    //--------------------------------------------------------------
    // check all vector <--> tensor <--> pixel etc methods and combinations
    void testCopyMethods() {
        // load image
        ofImage img_orig;
        img_orig.load("images/grace_hopper.png");   // jpgs don't work with tensorflow! :(

        int image_width = img_orig.getWidth();
        int image_height = img_orig.getHeight();
        int num_channels = img_orig.getPixels().getNumChannels();
        int num_bytes = img_orig.getPixels().getTotalBytes();

        // allocate images
        for(int i=0; i<kNumIterations; i++) img[i].allocate(image_width, image_height, OF_IMAGE_COLOR);

        img[0] = img_orig;

        bool do_memcpy = true;

        Tensor tensor;

        // bounce data around a bunch of different data types and size, see if data is preserved
        tensor = Tensor(DT_FLOAT, tensorflow::TensorShape({ num_channels, image_width, image_height}));
        msa::tf::pixels_to_tensor(img[0].getPixels(), tensor, do_memcpy);

        msa::tf::tensor_to_pixels(tensor, img[1].getPixels(), do_memcpy);
        img[1].update();

        tensor = Tensor(DT_FLOAT, tensorflow::TensorShape({ image_width / 2, image_height * num_channels * 2}));
        msa::tf::array_to_tensor(img[1].getPixels().getData(), tensor, do_memcpy);

        vector<float> v1;
        msa::tf::tensor_to_vector(tensor, v1, do_memcpy);
        assert(v1.size() == kInputElements);

        tensor = Tensor(DT_FLOAT, tensorflow::TensorShape({ num_bytes }));
        msa::tf::vector_to_tensor(v1, tensor, do_memcpy);

        msa::tf::tensor_to_array(tensor, img[2].getPixels().getData(), do_memcpy);
        img[2].update();

        tensor = Tensor(DT_FLOAT, tensorflow::TensorShape({ num_channels*16, image_height/16, image_width/32, 32}));
        msa::tf::image_to_tensor(img[2], tensor, do_memcpy);
        msa::tf::tensor_to_image(tensor, img[3], do_memcpy);

        tensor = Tensor(DT_FLOAT, tensorflow::TensorShape({ num_channels, image_width, image_height }));
        msa::tf::image_to_tensor(img[3], tensor, do_memcpy);

        // CLEAR ALL IMAGES TO TEST AUTO ALLOCATIONS
        for(int i=4; i<kNumIterations; i++) img[i].clear();

        msa::tf::tensor_to_image(tensor, img[4], do_memcpy);

        tensor = Tensor(DT_FLOAT, tensorflow::TensorShape({ num_channels, image_width*2, image_height/2 }));
        msa::tf::image_to_tensor(img[4], tensor, do_memcpy);

        // it's normal that this looks mangled, it's not supposed to be resized, just reshaped
        msa::tf::tensor_to_image(tensor, img[5], do_memcpy);

        tensor = Tensor(DT_FLOAT, tensorflow::TensorShape({ num_channels, image_width, image_height }));
        msa::tf::image_to_tensor(img[5], tensor, do_memcpy);

        msa::tf::tensor_to_image(tensor, img[6], do_memcpy);

        // convert to grayscale
        ofImage img_temp;
        img_temp= img[6];
        img_temp.setImageType(OF_IMAGE_GRAYSCALE);
        img[7] = img_temp;

        tensor = Tensor(DT_FLOAT, tensorflow::TensorShape({ image_width, image_height }));
        msa::tf::image_to_tensor(img[7], tensor, do_memcpy);

        msa::tf::tensor_to_image(tensor, img[8], do_memcpy);

        tensor = Tensor(DT_FLOAT, tensorflow::TensorShape({ image_width/2, image_height*2 }));
        msa::tf::image_to_tensor(img[8], tensor, do_memcpy);

        msa::tf::tensor_to_image(tensor, img[9], do_memcpy);

        tensor = Tensor(DT_FLOAT, tensorflow::TensorShape({ image_width, image_height }));
        msa::tf::image_to_tensor(img[9], tensor, do_memcpy);

        msa::tf::tensor_to_image(tensor, img[10], do_memcpy);
    }

    //--------------------------------------------------------------
    void setup() {
        ofSetColor(255);
        ofBackground(0);
        ofSetVerticalSync(true);

        testCopyMethods();
    }

    //--------------------------------------------------------------
    void draw() {
        int x=0, y=0;
        for(int i=0; i<kNumIterations; i++) {
            if(x + img[i].getWidth() >= ofGetWidth()) {
                x = 0;
                y += img[i].getHeight();
            }
            img[i].draw(x, y);
            x += img[i].getWidth();
        }
    }
};

//========================================================================
int main() {
    ofSetupOpenGL(1200, 960, OF_WINDOW);
    ofRunApp(new ofApp());
}
