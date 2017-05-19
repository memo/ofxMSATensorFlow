/*
pix2pix (Image-to-Image Translation with Conditional Adversarial Nets).

An accessible explanation can be found [here](https://phillipi.github.io/pix2pix/) and [here](https://affinelayer.com/pix2pix/).

The network basically learns to map from one image to another.

E.g. in the example you draw in the left viewport, and it generates the image in the right viewport.

I'm supplying three pretrained models from the original paper: cityscapes, building facades, and maps.

Models are trained and saved in python with [this code](https://github.com/memo/pix2pix-tensorflow)
(which is based on [this](https://github.com/affinelayer/pix2pix-tensorflow) tensorflow implementation,
which is based on the original [torch implementation](https://phillipi.github.io/pix2pix/)),
and loaded in openframeworks for prediction.
 */

#include "ofMain.h"
#include "ofxMSATensorFlow.h"
#include "ofxOpenCv.h"



//--------------------------------------------------------------
//--------------------------------------------------------------
class ofApp : public ofBaseApp {
public:
    // a bunch of properties of the models
    // ideally should read from disk and vary with the model
    // but trying to keep the code minimal so hardcoding them
    const int input_shape[2] = {256, 256}; // dimensions {height, width} for input image
    const int output_shape[2] = {256, 256}; // dimensions {height, width} for output image
    const float input_range[2] = {-1, 1}; // range of values {min, max} that model expects for input
    const float output_range[2] = {-1, 1}; // range of values {min, max} that model outputs
    const string input_op_name = "generator/generator_inputs"; // name of op to feed input to
    const string output_op_name = "generator/generator_outputs"; // name of op to fetch output from


    // fbo for drawing into (will be fed to model)
    ofFbo fbo;

    // images in and out of model
    ofFloatImage img_in; // input to the model (read from fbo), copied to t_in
    ofFloatImage img_out; // output from the model (copied from t_outs)

    // tensors in and out of model (converted to and from the above images)
    tensorflow::Tensor t_in;   // input to the model
    vector<tensorflow::Tensor> t_outs; // all outputs of the model, note that this is a std::vector, there can be multiple outs

    // shared pointer to tensorflow::Session
    msa::tf::Session_ptr session;


    // model file management
    string model_root_dir = "models";
    vector<string> model_names;
    int cur_model_index = 0;


    // color management for drawing
    const int max_colors = 10; // max number of colors to extract from test image
    const int palette_draw_size = 50;
    vector<ofColor> colors; // contains color palette that we should use for drawing (extracted from test image)
    int draw_color_index = 0;
    ofColor draw_color;


    // other vars
    bool do_auto_run = true;    // auto run every frame
    int draw_mode = 0;     // draw lines vs boxes
    int draw_radius = 10;
    ofVec2f mousePressPos;



    //--------------------------------------------------------------
    void setup() {
        ofSetColor(255);
        ofBackground(0);
        ofSetVerticalSync(true);
        ofSetLogLevel(OF_LOG_VERBOSE);
        ofSetFrameRate(60);


        // scan models dir
        ofDirectory dir;
        dir.listDir(model_root_dir);
        if(dir.size()==0) {
            ofLogError() << "Could not find models folder. Did you download the data files and place them in the data folder? ";
            ofLogError() << "Download from https://github.com/memo/ofxMSATensorFlow/releases";
            ofLogError() << "More info at https://github.com/memo/ofxMSATensorFlow/wiki";
            assert(false);
            ofExit(1);
        }
        for(int i=0; i<dir.getFiles().size(); i++) model_names.push_back(dir.getName(i));
        sort(model_names.begin(), model_names.end());
        load_model_index(0);
    }



    //--------------------------------------------------------------
    // Load graph (model trained in and exported from python) by folder INDEX, and initialize session
    void load_model_index(int index) {
        cur_model_index = ofClamp(index, 0, model_names.size()-1);
        load_model(model_root_dir + "/" + model_names[cur_model_index]);
    }



    //--------------------------------------------------------------
    // Extract a color palette from an image using the K-means algorithm.
    // based on https://github.com/mantissa/ofxColorQuantizer by Jeremy Rotsztain
    // in turn based on ocvColorQuantize demo that ships with Cinder
    // https://github.com/cinder/Cinder-OpenCV/blob/master/samples/ocvColorQuantize
    static vector<ofColor> get_colors(ofPixels inputImage, int colorCount){
        const int sampleCount = inputImage.getHeight() * inputImage.getWidth();
        cv::Mat colorSamples( sampleCount, 1, CV_32FC3 );

        unsigned char * pixels = inputImage.getData();

        cv::MatIterator_<cv::Vec3f> sampleIt = colorSamples.begin<cv::Vec3f>();
        for(int i=0; i<sampleCount; i++){
            int pos = i * 3;
            *sampleIt = cv::Vec3f( pixels[pos], pixels[pos+1], pixels[pos+2] );
            sampleIt++;
        }

        cv::Mat labels, clusters;
        cv::kmeans( colorSamples, colorCount, labels, cv::TermCriteria(), 2, cv::KMEANS_RANDOM_CENTERS, clusters ); //cv::TermCriteria::COUNT, 8, 0

        vector<ofColor> colors;
        for( int i = 0; i < colorCount; ++i ){
            ofColor clusterColor = ofColor( clusters.at<cv::Vec3f>(i,0)[0], clusters.at<cv::Vec3f>(i,0)[1], clusters.at<cv::Vec3f>(i,0)[2] );
            colors.push_back(clusterColor);
        }

        return colors;
    }



    //--------------------------------------------------------------
    // Load graph (model trained in and exported from python) by folder NAME, and initialize session
    void load_model(string model_dir) {
        ofLogVerbose() << "loading model " << model_dir;
        // init session with graph
        session = msa::tf::create_session_with_graph(model_dir + "/graph_frz.pb");

        if(!session) {
            ofLogError() << "Could not initialize session. Did you download the data files and place them in the data folder? ";
            ofLogError() << "Download from https://github.com/memo/ofxMSATensorFlow/releases";
            ofLogError() << "More info at https://github.com/memo/ofxMSATensorFlow/wiki";
            assert(false);
            ofExit(1);
        }


        // init tensor for input, shape should be: {batch size, image height, image width, number of channels}
        t_in = tensorflow::Tensor(tensorflow::DT_FLOAT, {1, input_shape[0], input_shape[1], 3});

        // allocate fbo and images with correct dimensions, and no alpha channel
        ofLogVerbose() << "allocating fbo and images " << input_shape;
        fbo.allocate(input_shape[1], input_shape[0], GL_RGB);
        img_in.allocate(input_shape[1], input_shape[0], OF_IMAGE_COLOR);
        img_out.allocate(output_shape[1], output_shape[0], OF_IMAGE_COLOR);

        // load test image (also needed for color palette extraction)
        ofLogVerbose() << "loading test image";
        ofImage img;
        img.load(model_dir + "/test_image.png");
        if(img.isAllocated()) {
            fbo.begin();
            ofSetColor(255);
            img.draw(0, 0, fbo.getWidth(), fbo.getHeight());
            fbo.end();

            // get colors
            colors = get_colors(img.getPixels(), max_colors);
            draw_color_index = 0;
            if(colors.size() > 0) draw_color = colors[0];
        } else {
            ofLogError() << "Test image not found";
        }
    }



    //--------------------------------------------------------------
    // run model on an image
    void run_model(ofFloatPixels& pix) {
        // dump input image into tensor
        msa::tf::pixels_to_tensor(pix, t_in);

        // our input image is in the range 0...1 but we need to convert to input_range (e.g. -1...1)
        // probably is an API for doing this without a manual loop, but can't find it right now
        // e.g. this doesn't compile: t_in = t_in * (input_range[1] - input_range[0]) + input_range[0];
        // so doing it manual ghetto style
        auto t_data = t_in.flat<float>().data();
        for(int i=0; i<t_in.NumElements(); i++) t_data[i] = ofMap(t_data[i], 0.0f, 1.0f, input_range[0], input_range[1]);


        // run graph, feed inputs (t_in), fetch output (t_outs)
        // remember that t_outs is a std::vector, t_outs[0] contains the actual image data
        // also remember that t_outs[0] is going to be { batch size (==1), image height, image width, number of channels }
        vector<string> fetch_tensors = { output_op_name };
        tensorflow::Status status = session->Run({ { input_op_name, t_in }}, fetch_tensors, {}, &t_outs);

        if(status != tensorflow::Status::OK()) {
            ofLogError() << status.error_message();
            return;
        }

        // convert model output from tensor to image
        if(t_outs.size() > 0) {
            // output image is in the range output_range (e.g. -1...1) but we need to convert to 0...1
            // probably is an API for doing this without a loop, but can't find it right now
            // so doing it manual ghetto style
            auto t_data = t_outs[0].flat<float>().data();
            for(int i=0; i<t_outs[0].NumElements(); i++) t_data[i] = ofMap(t_data[i], output_range[0], output_range[1], 0.0f, 1.0f);

            // dump tensor into an ofImage
            msa::tf::tensor_to_image(t_outs[0], img_out);
        }
    }



    //--------------------------------------------------------------
    // draw image or fbo etc with border and label
    // typename T must have draw(x,y), isAllocated(), getWidth(), getHeight()
    template <typename T>
    bool drawImage(const T& img, string label) {
        if(img.isAllocated()) {
            ofSetColor(255);
            ofFill();
            img.draw(0, 0);

            // draw border
            ofNoFill();
            ofSetColor(200);
            ofSetLineWidth(1);
            ofDrawRectangle(0, 0, img.getWidth(), img.getHeight());

            // draw label
            ofDrawBitmapString(label, 10, img.getHeight()+15);

            ofTranslate(img.getWidth(), 0);
            return true;
        }

        return false;
    }



    //--------------------------------------------------------------
    void draw() {
        // read from fbo into img_in
        fbo.readToPixels(img_in.getPixels());
//        img_in.update(); // update so we can draw if need be (useful for debuging)

        // run model on it
        if(session) {
            if(do_auto_run) {
                run_model(img_in.getPixels());
            }
        }


        // DISPLAY STUFF

        stringstream str;
        str << ofGetFrameRate() << endl;
        str << endl;
        str << "ENTER : toggle auto run " << (do_auto_run ? "(X)" : "( )") << endl;
        str << "DEL   : clear drawing " << endl;
        str << "d     : toggle draw mode " << (draw_mode==0 ? "(lines)" : "(boxes)") << endl;
        str << "[/]   : change draw radius (" << draw_radius << ")" << endl;
        str << "-/+   : change draw color " << endl;
        str << "i     : get color from mouse" << endl;
        str << endl;
        str << "draw in the box on the left" << endl;
        str << "or drag an image (PNG) into it" << endl;

        str << endl;
        str << "Press number key to load model: " << endl;
        str << endl;

        for(int i=0; i<model_names.size(); i++) {
            auto marker = (i==cur_model_index) ? ">" : " ";
            str << " " << (i+1) << " : " << marker << " " << model_names[i] << endl;
        }
        str << endl;



        ofPushMatrix();
        {
            if(!drawImage(fbo, "fbo (draw in here)") ) str << "fbo not allocated !!" << endl;
//            if(!drawImage(img_in, "img_in") ) str << "img_in not allocated !!" << endl; // just to check fbo is reading correctly
            if(!drawImage(img_out, "img_out") ) str << "img_out not allocated !!" << endl;

            ofTranslate(20, 0);

            // draw texts
            ofSetColor(150);
            ofDrawBitmapString(str.str(), 0, 20);
        }
        ofPopMatrix();


        // draw colors
        ofFill();
        int x=0;
        int y=fbo.getHeight() + 30;

        // draw current color
        ofSetColor(draw_color);
        ofDrawCircle(x+palette_draw_size/2, y+palette_draw_size/2, palette_draw_size/2);
        ofSetColor(200);
        ofDrawBitmapString("current color", x+palette_draw_size+10, y+palette_draw_size/2);
        y += palette_draw_size + 10;

        // draw color palette
        for(int i=0; i<colors.size(); i++) {
            ofSetColor(colors[i]);
            ofDrawCircle(x + palette_draw_size/2, y + palette_draw_size/2, palette_draw_size/2);

            // draw outline if selected color
            if(colors[i] == draw_color) {
                ofPushStyle();
                ofNoFill();
                ofSetColor(255);
                ofSetLineWidth(3);
                ofDrawRectangle(x, y, palette_draw_size, palette_draw_size);
                ofPopStyle();
            }

            x += palette_draw_size;

            // wrap around if doesn't fit on screen
            if(x > ofGetWidth() - palette_draw_size) {
                x = 0;
                y += palette_draw_size;
            }
        }


        // display drawing helpers
        ofSetColor(draw_color);
        ofNoFill();
        switch(draw_mode) {
        case 0:
            ofDrawCircle(ofGetMouseX(), ofGetMouseY(), draw_radius);
            break;
        case 1:
            if(ofGetMousePressed(0)) {
                ofDrawRectangle(mousePressPos.x, mousePressPos.y, ofGetMouseX()-mousePressPos.x, ofGetMouseY()-mousePressPos.y);
            }
        }
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

        case 'd':
        case 'D':
            draw_mode = 1 - draw_mode;
            break;

        case '[':
            if(draw_radius > 0) draw_radius--;
            break;

        case ']':
            draw_radius++;
            break;

        case '-':
        case '_':
            draw_color_index--;
            if(draw_color_index < 0) draw_color_index += colors.size(); // wrap around
            draw_color = colors[draw_color_index];
            break;

        case '=':
        case '+':
            draw_color_index++;
            if(draw_color_index >= colors.size()) draw_color_index -= colors.size(); // wrap around
            draw_color = colors[draw_color_index];
            break;

        case 'i':
        case 'I':
            if(ofGetMouseX() < fbo.getWidth() && ofGetMouseY() < fbo.getHeight()) {
                draw_color = img_in.getColor(ofGetMouseX(), ofGetMouseY());
            }
            break;

        case OF_KEY_DEL:
        case OF_KEY_BACKSPACE:
            fbo.begin();
            ofClear(0);
            fbo.end();
            break;

        case OF_KEY_RETURN:
            do_auto_run ^= true;
            break;
        }
    }



    //--------------------------------------------------------------
    void mouseDragged( int x, int y, int button) {
        switch(draw_mode) {
        case 0: // draw lines
            fbo.begin();
            ofSetColor(draw_color);
            ofFill();
            ofDrawCircle(x, y, draw_radius);
            ofSetLineWidth(draw_radius*2);
            ofDrawLine(x, y, ofGetPreviousMouseX(), ofGetPreviousMouseY());
            fbo.end();
            break;
        case 1: // draw boxes
            break;
        }
    }



    //--------------------------------------------------------------
    void mousePressed( int x, int y, int button) {
        mousePressPos = ofVec2f(x, y);
        mouseDragged(x, y, button);
    }



    //--------------------------------------------------------------
    virtual void mouseReleased(int x, int y, int button) {
        switch(draw_mode) {
        case 0: // draw lines
            break;
        case 1: // draw boxes
            fbo.begin();
            ofSetColor(draw_color);
            ofFill();
            ofDrawRectangle(mousePressPos.x, mousePressPos.y, x-mousePressPos.x, y-mousePressPos.y);
            fbo.end();
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
        if(img.isAllocated()) {
            fbo.begin();
            ofSetColor(255);
            img.draw(0, 0, fbo.getWidth(), fbo.getHeight());
            fbo.end();
        }
    }


};

//========================================================================
int main() {
    ofSetupOpenGL(800, 450, OF_WINDOW);
    ofRunApp(new ofApp());
}
