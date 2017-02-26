/*
Generative handwriting with a Long Short-Term Memory (LSTM) Recurrent Mixture Density Network (RMDN),
ala [Graves2013](https://arxiv.org/abs/1308.0850)

Brilliant tutorial on inner workings [here](http://blog.otoro.net/2015/12/12/handwriting-generation-demo-in-tensorflow/),
which also provides the base for the training code.

Models are trained and saved in python with this code (https://github.com/memo/write-rnn-tensorflow),
and loaded in openframeworks for prediction.
Given a sequence of points, the model predicts the position for the next point and pen-up probability.

I'm supplying a model pretrained on the [IAM online handwriting dataset](http://www.fki.inf.unibe.ch/databases/iam-on-line-handwriting-database)

Note that this demo does not do handwriting *synthesis*, i.e. text to handwriting ala [Graves' original demo](https://www.cs.toronto.edu/~graves/handwriting.html)
It just does *asemic* handwriting, producing sguiggles that are statistically similar to the training data,
e.g. same kinds of slants, curvatures, sharpnesses etc., but not nessecarily legible.

There is an implementation (and great tutorial) of *synthesis* using attention [here](https://greydanus.github.io/2016/08/21/handwriting/)
which I am also currently converting to work in openframeworks.
This attention-based synthesis implementation is also based on [Graves2013](https://arxiv.org/abs/1308.0850),
which I highly recommend to anyone really interested in understanding generative RNNs
 */

#include "ofMain.h"
#include "ofxMSATensorFlow.h"


//--------------------------------------------------------------
// sample a point from a bivariate gaussian mixture model
ofVec2f sample_from_bi_gmm(std::default_random_engine& rng,// random number generator
                           const vector<float>& o_pi,      // vector of mixture weights
                           const vector<float>& o_mu1,     // means 1
                           const vector<float>& o_mu2,     // means 2
                           const vector<float>& o_sigma1,  // sigmas 1
                           const vector<float>& o_sigma2,  // sigmas 2
                           const vector<float>& o_corr     // correlations
                           ) {

    ofVec2f ret;

    int k = o_pi.size();
    if(k == 0 || o_mu1.size() != k || o_mu2.size() != k || o_sigma1.size() != k || o_sigma2.size() != k || o_corr.size() != k) {
        ofLogWarning() << " sample_from_bi_gmm vector size mismatch ";
        return ret;
    }

    // pick mixture component to sample from
    int i = msa::tf::sample_from_prob(rng, o_pi);

    // sample i'th bivariate gaussian
    double mu1 = o_mu1[i];
    double mu2 = o_mu2[i];
    double sigma1 = o_sigma1[i];
    double sigma2 = o_sigma2[i];
    double corr = o_corr[i];

    // two independent zero mean, unit variance gaussian variables
    std::normal_distribution<double> gaussian(0.0, 1.0);

    double n1 = gaussian(rng);
    double n2 = gaussian(rng);

    ret.x = mu1 + sigma1 * n1;
    ret.y = mu2 + sigma2 * (corr * n1 + sqrt(1 - corr*corr) *n2);

    return ret;
}


//--------------------------------------------------------------
// visualise bivariate gaussian distribution
void draw_bi_gaussian(float mu1,     // mean 1
                      float mu2,     // mean 2
                      float sigma1,  // sigma 1
                      float sigma2,  // sigma 2
                      float corr    // correlation
                      ) {
    ofDrawEllipse(mu1, mu2, sigma1, sigma2);
}



//--------------------------------------------------------------
// visualise bivariate gaussian mixture model
void draw_bi_gmm(const vector<float>& o_pi,      // vector of mixture weights
                 const vector<float>& o_mu1,     // means 1
                 const vector<float>& o_mu2,     // means 2
                 const vector<float>& o_sigma1,  // sigmas 1
                 const vector<float>& o_sigma2,  // sigmas 2
                 const vector<float>& o_corr,    // correlations
                 const ofVec2f& offset=ofVec2f(0, 0),
                 float scale=1.0
        ) {

    int k = o_pi.size();
    if(k == 0 || o_mu1.size() != k || o_mu2.size() != k || o_sigma1.size() != k || o_sigma2.size() != k || o_corr.size() != k) {
        ofLogWarning() << " draw_gmm vector size mismatch ";
        return;
    }

    ofPushMatrix();
    ofTranslate(offset);
    ofScale(scale, scale);
    for(int i=0; i<k; i++) {
        float alpha = ofLerp(0.2, 1.0, o_pi[i]);
        ofSetColor(255, 0, 0, 255 * alpha);
        draw_bi_gaussian(o_mu1[i], o_mu2[i], o_sigma1[i], o_sigma2[i], o_corr[i]);
    }
    ofPopMatrix();
}


//--------------------------------------------------------------
//--------------------------------------------------------------
class ofApp : public ofBaseApp {
public:

    // shared pointer to tensorflow::Session
    msa::tf::Session_ptr session;


    // data in and out of model
    tensorflow::Tensor t_data_in;   // data in
    tensorflow::Tensor t_state;     // current lstm state
    vector<tensorflow::Tensor> t_out; // returned from session run [data_out_pi, data_out_mu1, data_out_mu2, data_out_sigma1, data_out_sigma2, data_out_corr, data_out_eos, state_out]

    // convert data in t_out convert to more managable types
    vector<float> o_pi;     // contains all mixture weights for (bivariate) gaussian mixture model, output by network (e.g. default 20 components)
    vector<float> o_mu1;    // " means 1
    vector<float> o_mu2;    // " means 2
    vector<float> o_sigma1; // " sigmas 1
    vector<float> o_sigma2; // " sigmas 2
    vector<float> o_corr;   // " correlations
    float o_eos;    // end of stroke probability

    // stores pts. xy storing relative offset from prev pos, and z storing end of stroke (0: draw, 1: eos)
    vector<ofVec3f> pts;


    // model file management
    string model_root_dir = "models";
    vector<string> model_names;
    int cur_model_index = 0;


    // random generator for sampling
    std::default_random_engine rng;


    // other vars
    int prime_length = 300;
    float draw_scale = 5;
    ofVec2f draw_pos = ofVec2f(100, 50);

    bool do_auto_run = true;    // auto run every frame
    bool do_run_once = false;   // only run one frame


    //--------------------------------------------------------------
    void setup() {
        //        ofBackground(0);
        ofSetVerticalSync(true);
        ofSetFrameRate(60);
        ofSetLogLevel(OF_LOG_VERBOSE);
        //        ofBackground(220);


        // scan models dir
        ofDirectory dir;
        dir.listDir(model_root_dir);
        for(int i=0; i<dir.getFiles().size(); i++) model_names.push_back(dir.getName(i));

        load_model_index(0);

        // seed rng
        rng.seed(ofGetSystemTimeMicros());
    }


    //--------------------------------------------------------------
    // Load graph (i.e. trained model and exported  from python) by folder index
    // and initialize session
    void load_model_index(int index) {
        cur_model_index = ofClamp(index, 0, model_names.size()-1);
        load_model(model_root_dir + "/" + model_names[cur_model_index]);
    }


    //--------------------------------------------------------------
    // Load graph (i.e. trained model and exported  from python) by folder name
    // and initialize session
    void load_model(string dir) {
        // init session with graph
        session = msa::tf::create_session_with_graph(dir + "/graph_frz.pb");

        // init tensor for input
        // meeds to be 3 floats (x, y, end of stroke), BUT not a vector, but rank3 tensor with other dims 1.
        // WHY? because that's how the model was designed to make the internal calculations easier (batch size etc)
        // tbh the model could be redesigned to accept just a 3 element vector, and then internally shift up to 1x1x3
        t_data_in = tensorflow::Tensor(tensorflow::DT_FLOAT, {1, 1, 3});

        // prime model
        prime_model(pts, prime_length);
    }


    //--------------------------------------------------------------
    // prime model with string
    void prime_model(const vector<ofVec3f>& prime_data, int prime_length) {
        t_state = tensorflow::Tensor(); // reset initial state to use zeros
        for(int i=MAX(0, prime_data.size()-prime_length); i<prime_data.size(); i++) {
            run_model(prime_data[i], t_state);
        }

    }



    //--------------------------------------------------------------
    // run model with one character
    void run_model(ofVec3f pt, const tensorflow::Tensor &state_in = tensorflow::Tensor()) {
        // format input data
        msa::tf::array_to_tensor(pt.getPtr(), t_data_in);

        // run graph, feed inputs, fetch output
        vector<string> fetch_tensors = { "data_out_pi", "data_out_mu1", "data_out_mu2", "data_out_sigma1", "data_out_sigma2", "data_out_corr", "data_out_eos", "state_out" };
        tensorflow::Status status;
        if(state_in.NumElements() > 0) {
            status = session->Run({ { "data_in", t_data_in }, { "state_in", state_in } }, fetch_tensors, {}, &t_out);
        } else {
            status = session->Run({ { "data_in", t_data_in }}, fetch_tensors, {}, &t_out);
        }

        if(status != tensorflow::Status::OK()) {
            ofLogError() << status.error_message();
            return;
        }

        if(t_out.size() > 1) {
            o_pi        = msa::tf::tensor_to_vector<float>(t_out[0]);
            o_mu1       = msa::tf::tensor_to_vector<float>(t_out[1]);
            o_mu2       = msa::tf::tensor_to_vector<float>(t_out[2]);
            o_sigma1    = msa::tf::tensor_to_vector<float>(t_out[3]);
            o_sigma2    = msa::tf::tensor_to_vector<float>(t_out[4]);
            o_corr      = msa::tf::tensor_to_vector<float>(t_out[5]);
            o_eos       = msa::tf::tensor_to_scalar<float>(t_out[6]);

            t_state = t_out.back();
        }
    }


    //--------------------------------------------------------------
    void draw() {
        draw_pos.y = ofGetHeight()/2;

        stringstream str;
        str << ofGetFrameRate() << endl;
        str << endl;
        str << "DEL   : clear drawing " << endl;
        str << "BKSPE : delete last point " << endl;
        str << "TAB   : auto run (" << do_auto_run << ")" << endl;
        str << "RIGHT : sample one pt " << endl;
        str << endl;

        str << "Press number key to load model: " << endl;
        for(int i=0; i<model_names.size(); i++) {
            auto marker = (i==cur_model_index) ? ">" : " ";
            str << " " << (i+1) << " : " << marker << " " << model_names[i] << endl;
        }

        str << endl;
        str << "Draw with mouse to prime the model" << endl;
        str << endl;


        if(session) {
            // sample 2d position from bivariate gaussian mixture model
            ofVec2f pt_pos = sample_from_bi_gmm(rng, o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr);

            ofVec3f pt = pt_pos;

            pt.z = ofRandomuf() < o_eos;

            if(do_auto_run || do_run_once) {
                if(do_run_once) do_run_once = false;

                // add sampled pt to drawing
                pts.push_back(pt);

                // feed sampled pt back into model
                run_model(pt, t_state);
            }
        }

        // construct pts with absolute positions (not relative), scaled and positioned
        vector<ofVec3f> pts_real(pts.size()+1);
        pts_real[0] = ofVec3f(draw_pos);
        for(int i=0; i<pts.size(); i++) {
            pts_real[i+1] = pts_real[i] + pts[i] * draw_scale;
            pts_real[i+1].z = pts[i].z;   // z component is eos state
        }

        ofVec2f last_pt_real = pts_real.back();

        // display probabilities
        draw_bi_gmm(o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, last_pt_real, draw_scale);

        // draw points. Treats x,y as position, and z as end of stroke
        ofSetColor(0);
        ofSetLineWidth(3);
        for(int i=0; i<pts_real.size()-1; i++) {
            // draw line if not eos
            if(pts_real[i].z < 0.5) {
                ofVec2f p0 = pts_real[i];
                ofVec2f p1 = pts_real[i+1];
                ofDrawLine(p0, p1);
            }
        }

        // if writing goes off screen, clear drawing
        if(last_pt_real.x > ofGetWidth() || last_pt_real.y > ofGetHeight() || last_pt_real.y < 0) {
            pts.clear();
        }


        // draw texts
        ofSetColor(100);
        ofDrawBitmapString(str.str(), 20, 20);
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

        case OF_KEY_DEL:
            pts.clear();
            break;

        case OF_KEY_BACKSPACE:
            pts.pop_back();
            //            prime_model(pts, prime_length); // prime model on key release to avoid lockup if key is held down
            do_auto_run = false;
            break;

        case OF_KEY_TAB:
            do_auto_run ^= true;
            break;

        case OF_KEY_RIGHT:
            do_run_once = true;
            do_auto_run = false;
            break;

        default:
            break;
        }
    }


    //--------------------------------------------------------------
    void keyReleased(int key) {
        switch(key) {
        case OF_KEY_BACKSPACE:
            prime_model(pts, prime_length); // prime model on key release to avoid lockup if key is held down
            break;
        }

    }

};

//========================================================================
int main() {
    ofSetupOpenGL(1280, 720, OF_WINDOW);
    ofRunApp(new ofApp());
}
