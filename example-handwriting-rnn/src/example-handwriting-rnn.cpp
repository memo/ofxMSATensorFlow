/*
Generative handwriting with Long Short-Term Memory (LSTM) Recurrent Mixture Density Network (RMDN),
ala [Graves2013](https://arxiv.org/abs/1308.0850).

Brilliant tutorial on inner workings [here](http://blog.otoro.net/2015/12/12/handwriting-generation-demo-in-tensorflow/),
which also provides the base for the training code (also see javscript port and tutorial [here](http://distill.pub/2016/handwriting/)).

Models are trained and saved in python with [this code](https://github.com/memo/write-rnn-tensorflow),
and loaded in openframeworks for prediction.

Given a sequence of points, the model predicts the position for the next point and pen-up probability.

I'm supplying a model pretrained on the [IAM online handwriting dataset](http://www.fki.inf.unibe.ch/databases/iam-on-line-handwriting-database).

Note that this demo does not do handwriting *synthesis*, i.e. text to handwriting ala [Graves' original demo](https://www.cs.toronto.edu/~graves/handwriting.html).
It just does *asemic* handwriting, producing squiggles that are statistically similar to the training data,
e.g. same kinds of slants, curvatures, sharpnesses etc., but not nessecarily legible.
There is an implementation (and great tutorial) of *synthesis* using attention [here](https://greydanus.github.io/2016/08/21/handwriting/),
which I am also currently converting to work in openframeworks.
This attention-based synthesis implementation is also based on [Graves2013](https://arxiv.org/abs/1308.0850),
which I highly recommend to anyone really interested in understanding generative RNNs.
 */

#include "ofMain.h"
#include "ofxMSATensorFlow.h"

//--------------------------------------------------------------
//--------------------------------------------------------------
class ofApp : public ofBaseApp {
public:

    // shared pointer to tensorflow::Session
    msa::tf::Session_ptr session;

    // tensors in and out of model
    tensorflow::Tensor t_data_in;   // data in
    tensorflow::Tensor t_state;     // current lstm state
    vector<tensorflow::Tensor> t_out; // returned from session run [data_out_pi, data_out_mu1, data_out_mu2, data_out_sigma1, data_out_sigma2, data_out_corr, data_out_eos, state_out]

    // convert output of model (t_out) to more managable types
    // parameters of bivariate gaussian mixture model, and probability for end of stroke
    struct ProbData {
        vector<float> o_pi;      // contains all mixture weights (e.g. default 20 components)
        vector<float> o_mu1;     // " means 1
        vector<float> o_mu2;     // " means 2
        vector<float> o_sigma1;  // " sigmas 1
        vector<float> o_sigma2;  // " sigmas 2
        vector<float> o_corr;    // " correlations
        float o_eos;             // end of stroke probability
    };

    ProbData last_model_output;

    // all points. xy storing relative offset from prev pos, and z storing end of stroke (0: draw, 1: eos)
    vector<ofVec3f> pts;

    // probability parameters for each point for visualisation
    // i'th ProbData contains probability parameters which the i'th point (pts[i]) was sampled from
    vector<ProbData> probs;

    // vector of meshes where each mesh contains a ton of sampled points (to visualised the distribution)
    vector<ofVboMesh> prob_meshes;


    // model file management
    string model_root_dir = "models";
    vector<string> model_names;
    int cur_model_index = 0;


    // random generator for sampling
    std::default_random_engine rng;


    // other vars
    int prime_length = 300;
    float draw_scale = 5;
    ofVec2f draw_pos = ofVec2f(100, 20);
    int sample_vis_count = 2000; // lower this if it's running slow (number of samples per point for probability visualisation)
    float sample_vis_alpha = 0.05; // increase this if you lower the number above (alpha for each sample in probability visualisation)

    bool do_auto_run = true;    // auto run every frame
    bool do_run_once = false;   // only run one frame


    //--------------------------------------------------------------
    void setup() override {
        ofSetVerticalSync(true);
        ofSetLogLevel(OF_LOG_VERBOSE);
        ofSetFrameRate(60);
        ofBackground(220);

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

        // seed rng
        rng.seed(ofGetSystemTimeMicros());
    }


    //--------------------------------------------------------------
    void reset() {
        pts.clear();
        probs.clear();
        prob_meshes.clear();
        t_state = tensorflow::Tensor(); // reset state
    }


    //--------------------------------------------------------------
    // Load graph (model trained in and exported from python) by folder INDEX, and initialize session
    void load_model_index(int index) {
        cur_model_index = ofClamp(index, 0, model_names.size()-1);
        load_model(model_root_dir + "/" + model_names[cur_model_index]);
    }


    //--------------------------------------------------------------
    // Load graph (model trained in and exported from python) by folder NAME, and initialize session
    void load_model(string dir) {
        // init session with graph
        session = msa::tf::create_session_with_graph(dir + "/graph_frz.pb");

        if(!session) {
            ofLogError() << "Could not initialize session. Did you download the data files and place them in the data folder? ";
            ofLogError() << "Download from https://github.com/memo/ofxMSATensorFlow/releases";
            ofLogError() << "More info at https://github.com/memo/ofxMSATensorFlow/wiki";
            assert(false);
            ofExit(1);
        }

        // init tensor for input
        // meeds to be 3 floats (x, y, end of stroke).
        // HOWEVER input is not a vector, but a rank 3 tensor with shape {1, 1, 3}
        // WHY? because that's how the model was designed to make the internal calculations easier (batch size etc)
        // TBH the model could be redesigned to accept just a 3 element vector, and then internally reshaped, but I'm lazy
        t_data_in = tensorflow::Tensor(tensorflow::DT_FLOAT, {1, 1, 3});

        // prime model
        prime_model(pts, prime_length);
    }


    //--------------------------------------------------------------
    // prime model with a sequence of points
    // this runs the data through the model element by element, so as to update its internal state (stored in t_state)
    // next time we feed the model an element to make a prediction, it will make the prediction primed on this state (i.e. sequence of elements)
    void prime_model(const vector<ofVec3f>& prime_data, int prime_length) {
        t_state = tensorflow::Tensor(); // reset initial state to use zeros
        for(int i=MAX(0, prime_data.size()-prime_length); i<prime_data.size(); i++) {
            run_model(prime_data[i], t_state);
        }
    }



    //--------------------------------------------------------------
    // run model on a single point
    void run_model(ofVec3f pt, const tensorflow::Tensor &state_in = tensorflow::Tensor()) {
        // copy input data into tensor
        msa::tf::array_to_tensor(pt.getPtr(), t_data_in);

        // run graph, feed inputs, fetch output
        vector<string> fetch_tensors = { "data_out_pi", "data_out_mu1", "data_out_mu2", "data_out_sigma1", "data_out_sigma2", "data_out_corr", "data_out_eos", "state_out" };
        tensorflow::Status status;
        if(state_in.NumElements() > 0) {
            // use state_in if passed in as parameter
            status = session->Run({ { "data_in", t_data_in }, { "state_in", state_in } }, fetch_tensors, {}, &t_out);
        } else {
            // otherwise model will use internally init state to zeros
            status = session->Run({ { "data_in", t_data_in }}, fetch_tensors, {}, &t_out);
        }

        if(status != tensorflow::Status::OK()) {
            ofLogError() << status.error_message();
            return;
        }

        // convert model output from tensors to more manageable types
        if(t_out.size() > 1) {
            last_model_output.o_pi        = msa::tf::tensor_to_vector<float>(t_out[0]);
            last_model_output.o_mu1       = msa::tf::tensor_to_vector<float>(t_out[1]);
            last_model_output.o_mu2       = msa::tf::tensor_to_vector<float>(t_out[2]);
            last_model_output.o_sigma1    = msa::tf::tensor_to_vector<float>(t_out[3]);
            last_model_output.o_sigma2    = msa::tf::tensor_to_vector<float>(t_out[4]);
            last_model_output.o_corr      = msa::tf::tensor_to_vector<float>(t_out[5]);
            last_model_output.o_eos       = msa::tf::tensor_to_scalar<float>(t_out[6]);

            // save lstm state for next run
            t_state = t_out.back();
        }
    }


    //--------------------------------------------------------------
    void draw() override {
        draw_pos.y = ofGetHeight() * 0.4;

        stringstream str;
        str << ofGetFrameRate() << endl;
        str << endl;
        str << "ENTER : prime the model and toggle auto run " << (do_auto_run ? "(X)" : "( )") << endl;
        str << "RIGHT : sample one pt " << endl;
        str << "LEFT  : delete last point " << endl;
        str << "DEL   : clear drawing " << endl;
        str << endl;
        str << "Write or draw with mouse and then press ENTER to prime the model and carry on" << endl;
        str << "e.g. try writing/drawing very spikey vs round shapes, or slanting up vs down" << endl;
        str << "remember that the IAM model has only been trained on handwriting, so it will only produce handwriting, and not drawings!" << endl;
        str << "but maybe the handwriting it generates will have stylistic similarities to what you wrote or drew" << endl;
        str << endl;

        str << "Press number key to load model: " << endl;
        for(int i=0; i<model_names.size(); i++) {
            auto marker = (i==cur_model_index) ? ">" : " ";
            str << " " << (i+1) << " : " << marker << " " << model_names[i] << endl;
        }


        if(session) {
            // sample 2d position from bivariate gaussian mixture model
            ofVec3f pt = msa::tf::sample_from_bi_gmm(rng, last_model_output.o_pi, last_model_output.o_mu1, last_model_output.o_mu2, last_model_output.o_sigma1, last_model_output.o_sigma2, last_model_output.o_corr);

            // end of stroke probability
            pt.z = ofRandomuf() < last_model_output.o_eos;

            if(do_auto_run || do_run_once) {
                if(do_run_once) do_run_once = false;

                // add sampled pt to drawing
                pts.push_back(pt);

                // add probabilities to vector
                probs.push_back(last_model_output);

                // take many samples for probability visualisation
                ofVboMesh mesh;
                for(int i=0; i<sample_vis_count; i++) {
                    mesh.addVertex(ofVec3f(msa::tf::sample_from_bi_gmm(rng, last_model_output.o_pi, last_model_output.o_mu1, last_model_output.o_mu2, last_model_output.o_sigma1, last_model_output.o_sigma2, last_model_output.o_corr)));

                    // color most of the points blue, and some red, so should be redder towards dense parts of the distribution
                    mesh.addColor( i < sample_vis_count * 0.9 ? ofColor(0, 0, 255, sample_vis_alpha*255) : ofColor(255, 0, 0, sample_vis_alpha*255) );
                }
                prob_meshes.push_back(mesh);

                // feed sampled pt back into model
                run_model(pt, t_state);
            }
        }


        // draw stuff
        ofVec2f cur_pos = draw_pos;  // keep track of screen coords of last drawn point
        ofPushStyle();
        ofSetLineWidth(3);
        glPointSize(3); // is there no ofXXX function for this?
        ofPushMatrix();
        {
            ofTranslate(draw_pos);       // start drawing at draw_pos
            ofScale(draw_scale, draw_scale);

            for(int i=0; i<pts.size(); i++) {
                ofVec2f offset( pts[i] );

                ofTranslate(offset);

                // draw points. xy is relative position offset, z is end of stroke
                if(i>0 && pts[i-1].z < 0.5) { //  need to check z of previous stroke to decide whether to draw or not
                    ofSetColor(0);
                    ofDrawLine(ofVec2f::zero(), -offset);
                }

                // visualise probabilities
                ofPushMatrix();
                {
                    ofTranslate(0, ofGetHeight()/3/draw_scale);    // offset in y


                    // visualise probabilities via sample_vis_count # of samples per point (in BLUE-PURPLE-RED)
                    const auto& pm = prob_meshes[i];
                    pm.drawVertices();


                    // NOT USING THIS METHOD ANYMORE, THE ONE ABOVE IS MORE ACCURATE I THINK, AND NICER
                    // visualise probabilities via ellipses on correlation matrix (in GREEN)
                    // the way I'm doing this is rather inefficient: every frame the matrices for each of the components for each of the points is reconstructed
                    // instead, this could be cached to save computation, but i'm trying to keep this a short, simple example
                    //                    const auto& p = probs[i];
                    //                    draw_bi_gmm(p.o_pi, p.o_mu1, p.o_mu2, p.o_sigma1, p.o_sigma2, p.o_corr);


                    // draw actual sampled point (in BLACK)
                    //                    ofSetColor(0, 0, 0);
                    //                    ofDrawCircle(0, 0, 2.0f/draw_scale); // need to divide by draw_scale to get back to pixel units
                }
                ofPopMatrix();

                cur_pos += offset * draw_scale;
            }
        }
        ofPopMatrix();
        ofPopStyle();

        // if writing goes off screen, clear drawing
        if(cur_pos.x < 0 || cur_pos.x > ofGetWidth() || cur_pos.y > ofGetHeight() || cur_pos.y < 0) {
            reset();
        }


        // draw texts
        ofSetColor(100);
        ofDrawBitmapString(str.str(), 20, 20);
    }


    //--------------------------------------------------------------
    void keyPressed(int key) override {
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

        case OF_KEY_DEL:
            reset();
            break;

        case OF_KEY_RETURN:
            do_auto_run ^= true;
            if(do_auto_run) prime_model(pts, prime_length);
            break;

        case OF_KEY_RIGHT:
            do_run_once = true;
            do_auto_run = false;
            break;

        case OF_KEY_LEFT:
            if(!pts.empty()) pts.pop_back();
            if(!probs.empty()) probs.pop_back();
            if(!prob_meshes.empty()) prob_meshes.pop_back();
            do_auto_run = false;
            //            prime_model(pts, prime_length); // prime model on key release to avoid lockup if key is held down
            break;

        default:
            break;
        }
    }


    //--------------------------------------------------------------
    void keyReleased(int key) override {
        switch(key) {
        case OF_KEY_LEFT:
            prime_model(pts, prime_length); // prime model on key release to avoid lockup if key is held down
            break;
        }
    }


    //--------------------------------------------------------------
    void mousePressed(int x, int y, int button) override {
        if(!pts.empty()) pts.back().z = 1;  // last point should be end of stroke
        mouseDragged(x, y, button);
    }


    //--------------------------------------------------------------
    void mouseDragged(int x, int y, int button) override {
        ofVec2f pt(x, y);

        // transform screen pos to relative unscaled pos
        for(const auto& p : pts) pt -= ofVec2f( p ) * draw_scale;    // subtract relative offsets of previous points
        pt -= draw_pos;
        pt /= draw_scale;

        pts.push_back(pt);

        // add empty data so arrays are in sync
        probs.push_back(ProbData());
        prob_meshes.push_back(ofVboMesh());

        do_auto_run = false;
        do_run_once = false;
    }


    //--------------------------------------------------------------
    void mouseReleased(int x, int y, int button) override {
        if(!pts.empty()) pts.back().z = 1;  // last point should be end of stroke
    }


};

//========================================================================
int main() {
    ofSetupOpenGL(1280, 720, OF_WINDOW);
    ofRunApp(new ofApp());
}
