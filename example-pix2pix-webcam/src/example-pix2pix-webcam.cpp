/*
Same as pix2pix-example with the addition of live webcam input. See description of pix2pix-example for more info on pix2pix and the models I provide.

I'm using a very simple and ghetto method of transforming the webcam input into the desired colour palette before feeding into the model. See the code for more info on this.

 */

#include "ofMain.h"
#include "ofxMSATensorFlow.h"
#include "ofxOpenCv.h"


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


    // for video capture stuff
    ofVideoGrabber video_grabber;   // for capturing from camera
    ofImage img_capture;       // captured image cropped to (height, height)
    ofxCvColorImage cv_capture; // captured image as opencv image
    ofxCvGrayscaleImage cv_capture_gray; // grayscale opencv image
    ofImage img_processed;       // final processed image before writing to fbo
    bool capture_enabled = true;
    bool capture_flip_h = false;


    // fbos
    ofFbo fbo_draw; // for drawing into
    ofFbo fbo_comp; // composite of drawing and video input (will be fed to model)

    // images in and out of model
    // preallocating these to save allocating them every frame
    ofFloatImage img_in; // input to the model (read from fbo)
    ofFloatImage img_out; // output from the model

    // model file management
    ofDirectory models_dir;    // data/models folder which contains subfolders for each model
    int cur_model_index = 0; // which model (i.e. folder) we're currently using


    // color management for drawing
    vector<ofColor> colors; // contains color palette to be used for drawing (loaded from data/models/XYZ/palette.txt)
    const int palette_draw_size = 50;
    int draw_color_index = 0;
    ofColor draw_color;


    // other vars
    bool do_auto_run = true;    // auto run every frame
    int draw_mode = 0;     // draw vs boxes
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
        video_grabber.setup(640, 480);
    }


    //--------------------------------------------------------------
    // Load graph (model trained in and exported from python) by folder NAME, and initialise session
    void load_model(string model_dir) {
        ofLogVerbose() << "loading model " << model_dir;

        // init the model
        // note that it expects arrays for input op names and output op names, so just use {}
        model.setup(ofFilePath::join(model_dir, "graph_frz.pb"), {input_op_name}, {output_op_name});
        if(! model.is_loaded()) {
            ofLogError() << "Model init error.";
            ofLogError() << msa::tf::missing_data_error();
            assert(false);
            ofExit(1);
        }

        // init tensor for input. shape should be: {batch size, image height, image width, number of channels}
        // (ideally the SimpleModel graph loader would read this info from the graph_def and call this internally)
        model.init_inputs(tensorflow::DT_FLOAT, {1, input_shape[0], input_shape[1], 3});


        // allocate fbo and images with correct dimensions, and no alpha channel
        ofLogVerbose() << "allocating fbo and images " << input_shape;
        fbo_draw.allocate(input_shape[1], input_shape[0], GL_RGBA); // this one has alpha because it will be overlaid over the video input
        fbo_comp.allocate(input_shape[1], input_shape[0], GL_RGB); // this one doesn't have alpha
        img_in.allocate(input_shape[1], input_shape[0], OF_IMAGE_COLOR);
        img_out.allocate(output_shape[1], output_shape[0], OF_IMAGE_COLOR);


        // load test image
        ofLogVerbose() << "loading test image";
        ofImage img;
        img.load(ofFilePath::join(model_dir, "test_image.png"));
        if(img.isAllocated()) {
            fbo_comp.begin();
            ofSetColor(255);
            img.draw(0, 0, fbo_comp.getWidth(), fbo_comp.getHeight());
            fbo_comp.end();

        } else {
            ofLogError() << "Test image not found";
        }

        // load color palette for drawing
        ofLogVerbose() << "loading color palette";
        colors.clear();
        ofBuffer buf;
        buf = ofBufferFromFile(ofFilePath::join(model_dir, "/palette.txt"));
        if(buf.size()>0) {
            for(const auto& line : buf.getLines()) {
                ofLogVerbose() << line;
                if(line.size() == 6) // if valid hex code
                    colors.push_back(ofColor::fromHex(ofHexToInt(line)));
            }
            draw_color_index = 0;
            if(colors.size() > 0) draw_color = colors[0];
        } else {
            ofLogError() << "Palette info not found";
        }

        // load default brush info
        ofLogVerbose() << "loading default brush info";
        buf = ofBufferFromFile(ofFilePath::join(model_dir, "/default_brush.txt"));
        if(buf.size()>0) {
            auto str_info = buf.getFirstLine();
            ofLogVerbose() << str_info;
            auto str_infos = ofSplitString(str_info, " ", true, true);
            if(str_infos[0]=="draw") draw_mode = 0;
            else if(str_infos[0]=="box") draw_mode = 1;
            else ofLogError() << "Unknown draw mode: " << str_infos[0];

            draw_radius = ofToInt(str_infos[1]);

        } else {
            ofLogError() << "Default brush info not found";
        }

    }


    //--------------------------------------------------------------
    // Load model by folder INDEX
    void load_model_index(int index) {
        cur_model_index = ofClamp(index, 0, models_dir.size()-1);
        load_model(models_dir.getPath(cur_model_index));
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
        if(capture_enabled && video_grabber.isInitialized()) {
            // update video grabber (i.e. get next frame)
            video_grabber.update();
            if(video_grabber.isFrameNew()) {

                // crop center square and write to img_capture
                video_grabber.getPixels().cropTo(img_capture.getPixels(), (video_grabber.getWidth()-video_grabber.getHeight())/2, 0, video_grabber.getHeight(), video_grabber.getHeight());
                img_capture.update();

                // flip horizontally
                img_capture.mirror(false, capture_flip_h);

                // update opencv image cv_capture from cropped pixels
                cv_capture.setFromPixels(img_capture);

                // convert to grayscale
                cv_capture_gray = cv_capture;

                // blur a bit to remove noise
                cv_capture_gray.blur(3);

                // very ghetto way of finding edges (kind of) depending on draw_radius
                cvAdaptiveThreshold(cv_capture_gray.getCvImage(), cv_capture_gray.getCvImage(), 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, draw_radius*2+5, 2);
                cv_capture_gray.blur(draw_radius*2+1); // make sure blur amount is odd

                // very ghetto way of colouring a gray image
                // read from cv_capture_gray and write to img_processed using custom colour palette
                img_processed.allocate(cv_capture_gray.getWidth(), cv_capture_gray.getHeight(), OF_IMAGE_COLOR);    // will only allocate if dims don't match
                ofPixels& proc_pixels = img_processed.getPixels();
                const ofPixels& gray_pixels = cv_capture_gray.getPixels();
                for(int y=0; y<cv_capture_gray.getHeight(); y++) {
                    for(int x=0; x<cv_capture_gray.getWidth(); x++) {
                        float t = gray_pixels.getColor(x, y).r;
                        int color_index = round(ofMap(t, 0, 255, colors.size()-1, 0));
                        proc_pixels.setColor(x, y, colors[color_index]);
                    }
                }
                img_processed.update(); // update opengl texture with new pixels


                // draw to composite fbo
                fbo_comp.begin();
                ofSetColor(255);
                img_processed.draw(0, 0, fbo_comp.getWidth(), fbo_comp.getHeight());
                fbo_comp.end();
            }
        }

        // draw drawing into composite fbo
        fbo_comp.begin();
        ofSetColor(255);
        fbo_draw.draw(0, 0);
        fbo_comp.end();


        // read from composite fbo into img_in
        fbo_comp.readToPixels(img_in.getPixels());
        //        img_in.update(); // update so we can draw if need be (useful for debuging)

        // run model on it
        if(do_auto_run)
            model.run_image_to_image(img_in, img_out, input_range, output_range);

        // DISPLAY STUFF
        stringstream str;
        str << ofGetFrameRate() << endl;
        str << endl;
        str << "ENTER : toggle auto run " << (do_auto_run ? "(X)" : "( )") << endl;
        str << "DEL   : clear drawing " << endl;
        str << "v     : toggle capture " << (capture_enabled ? "(X)" : "( )") << endl;
        str << "h     : toggle horizontal flip" << (capture_flip_h ? "(X)" : "( )") << endl;
        str << "d     : toggle draw mode " << (draw_mode==0 ? "(draw)" : "(boxes)") << endl;
        str << "[/]   : change draw radius (" << draw_radius << ")" << endl;
        str << "z/x   : change draw color " << endl;
        str << "i     : get color from mouse" << endl;
        str << endl;
        str << "draw in the box on the left" << endl;
        str << "or drag an image (PNG) into it" << endl;
        str << endl;

        str << "Press number key to load model: " << endl;
        for(int i=0; i<models_dir.size(); i++) {
            auto marker = (i==cur_model_index) ? ">" : " ";
            str << " " << (i+1) << " : " << marker << " " << models_dir.getName(i) << endl;
        }


        ofPushMatrix();
        {
            if(!drawImage(fbo_comp, "fbo (draw in here)") ) str << "fbo not allocated !!" << endl;
            //            if(!drawImage(img_in, "img_in") ) str << "img_in not allocated !!" << endl; // just to check fbo is reading correctly
            if(!drawImage(img_out, "img_out") ) str << "img_out not allocated !!" << endl;

            ofTranslate(20, 0);

            // draw texts
            ofSetColor(150);
            ofDrawBitmapString(str.str(), 0, 20);
        }
        ofPopMatrix();


        // draw video input at bottom right at quarter size
        img_capture.draw(ofGetWidth()-img_capture.getWidth()/4, ofGetHeight()-img_capture.getHeight()/4, img_capture.getWidth()/4, img_capture.getHeight()/4);


        // draw colors
        ofFill();
        int x=0;
        int y=fbo_comp.getHeight() + 30;

        // draw current color
        ofSetColor(draw_color);
        ofDrawCircle(x+palette_draw_size/2, y+palette_draw_size/2, palette_draw_size/2);
        ofSetColor(200);
        ofDrawBitmapString("current draw color (change with z/x keys)", x+palette_draw_size+10, y+palette_draw_size/2);
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
        ofNoFill();
        switch(draw_mode) {
        case 0:
            ofSetLineWidth(3);
            ofSetColor(ofColor::black);
            ofDrawCircle(ofGetMouseX(), ofGetMouseY(), draw_radius+1);

            ofSetLineWidth(3);
            ofSetColor(draw_color);
            ofDrawCircle(ofGetMouseX(), ofGetMouseY(), draw_radius);

            break;
        case 1:
            if(ofGetMousePressed(0)) {
                ofSetLineWidth(3);
                ofSetColor(ofColor::black);
                ofDrawRectangle(mousePressPos.x-1, mousePressPos.y-1, ofGetMouseX()-mousePressPos.x+3, ofGetMouseY()-mousePressPos.y+3);

                ofSetLineWidth(3);
                ofSetColor(draw_color);
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
            draw_mode = 1 - draw_mode;
            break;

        case '[':
            if(draw_radius > 0) draw_radius--;
            break;

        case ']':
            draw_radius++;
            break;

        case 'z':
            draw_color_index--;
            if(draw_color_index < 0) draw_color_index += colors.size(); // wrap around
            draw_color = colors[draw_color_index];
            break;

        case 'x':
            draw_color_index++;
            if(draw_color_index >= colors.size()) draw_color_index -= colors.size(); // wrap around
            draw_color = colors[draw_color_index];
            break;

        case 'i':
        case 'I':
            if(ofGetMouseX() < fbo_draw.getWidth() && ofGetMouseY() < fbo_draw.getHeight()) {
                draw_color = img_in.getColor(ofGetMouseX(), ofGetMouseY());
            }
            break;

        case OF_KEY_DEL:
        case OF_KEY_BACKSPACE:
            fbo_draw.begin();
            ofClear(0);
            fbo_draw.end();

            fbo_comp.begin();
            ofClear(0);
            fbo_comp.end();
            break;

        case OF_KEY_RETURN:
            do_auto_run ^= true;
            break;
        }
    }



    //--------------------------------------------------------------
    void mouseDragged( int x, int y, int button) {
        switch(draw_mode) {
        case 0: // draw
            fbo_draw.begin();
            ofSetColor(draw_color);
            ofFill();
            if(draw_radius>0) ofDrawCircle(x, y, draw_radius);
            ofSetLineWidth(draw_radius*2+0.1f);
            ofDrawLine(x, y, ofGetPreviousMouseX(), ofGetPreviousMouseY());
            fbo_draw.end();
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
        case 0: // draw
            break;
        case 1: // boxes
            fbo_draw.begin();
            ofSetColor(draw_color);
            ofFill();
            ofDrawRectangle(mousePressPos.x, mousePressPos.y, x-mousePressPos.x, y-mousePressPos.y);
            fbo_draw.end();
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
            fbo_draw.begin();
            ofSetColor(255);
            img.draw(0, 0, fbo_draw.getWidth(), fbo_draw.getHeight());
            fbo_draw.end();
        }
    }


};

//========================================================================
int main() {
    ofSetupOpenGL(800, 450, OF_WINDOW);
    ofRunApp(new ofApp());
}
