/*
 * Simple example of char-rnn style LSTM trained in python, loaded and manipulated in openFrameworks
 * See github repo and wiki for more info
 *
 */

#include "ofMain.h"
#include "ofxMSATensorFlow.h"


//--------------------------------------------------------------
//--------------------------------------------------------------
class ofApp : public ofBaseApp {
public:

    // shared pointer to tensorflow::Session
    msa::tf::Session_ptr session;


    // for managing character <-> index mapping
    vector<char> int_to_char;
    map<int, char> char_to_int;


    // data in and out of model
    tensorflow::Tensor t_data_in;   // character index in
    tensorflow::Tensor t_state;     // current lstm state
    vector<tensorflow::Tensor> t_out; // returned from session run [ data_out (prob), state_out ]
    vector<float> probs;    // probabilities


    // generated text
    // managing word wrap in very ghetto way
    string text_full;
    list<string> text_lines = { "The" };
    int max_line_width = 120;
    int max_line_num = 50;


    // model file management
    string model_root_dir = "models";
    vector<string> model_names;
    int cur_model_index = 0;


    // random generator for sampling
    std::default_random_engine rng;


    // other vars
    int prime_length = 50;
    float sample_temp = 0.5f;

    bool do_auto_run = true;    // auto run every frame
    bool do_run_once = false;   // only run one character


    //--------------------------------------------------------------
    void setup() {
        ofSetColor(255);
        ofBackground(0);
        ofSetVerticalSync(true);
        ofSetFrameRate(60);
        ofSetLogLevel(OF_LOG_VERBOSE);


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

        // load character map
        load_chars(dir + "/chars.txt");

        // init tensor for input
        t_data_in = tensorflow::Tensor(tensorflow::DT_INT32, {1, 1});

        // prime model
        prime_model(text_full, prime_length);
    }


    //--------------------------------------------------------------
    // load character <-> index mapping
    void load_chars(string path) {
        ofLogVerbose() << "load_chars : " << path;
        int_to_char.clear();
        char_to_int.clear();
        ofBuffer buffer = ofBufferFromFile(path);

        for(auto line : buffer.getLines()) {
            char c = ofToInt(line); // TODO: will this manage unicode?
            int_to_char.push_back(c);
            int i = int_to_char.size()-1;
            char_to_int[c] = i;
            ofLogVerbose() << i << " : " << c;
        }
    }



    //--------------------------------------------------------------
    // prime model with string
    void prime_model(string prime_str, int prime_length) {
        t_state = tensorflow::Tensor(); // reset initial state to use zeros
        for(int i=MAX(0, prime_str.size()-prime_length); i<prime_str.size(); i++) {
            run_model(prime_str[i], t_state);
        }

    }



    //--------------------------------------------------------------
    // run model with one character
    void run_model(char ch, const tensorflow::Tensor &state_in = tensorflow::Tensor()) {
        // format input data
        msa::tf::scalar_to_tensor(char_to_int[ch], t_data_in);

        tensorflow::Status status;
        // run graph, feed inputs, fetch output
        if(state_in.NumElements() > 0) {
            status = session->Run({ { "data_in", t_data_in }, { "state_in", state_in } }, { "data_out", "state_out" }, {}, &t_out);
        } else {
            status = session->Run({ { "data_in", t_data_in }}, { "data_out", "state_out" }, {}, &t_out);
        }

        if(status != tensorflow::Status::OK()) {
            ofLogError() << status.error_message();
            return;
        }

        if(t_out.size() > 1) {
            probs = msa::tf::tensor_to_vector<float>(t_out[0]);
            probs = msa::tf::adjust_probs_with_temp(probs, sample_temp);
            t_state = t_out[1];
        }
    }


    //--------------------------------------------------------------
    void add_char(char ch) {
        // add sampled char to text
        if(ch == '\n') {
            text_lines.push_back("");
        } else {
            text_lines.back() += ch;
        }

        // ghetto word wrap
        if(text_lines.back().size() > max_line_width) {
            string text_line_cur = text_lines.back();
            text_lines.pop_back();
            auto last_word_pos = text_line_cur.find_last_of(" ");
            text_lines.push_back(text_line_cur.substr(0, last_word_pos));
            text_lines.push_back(text_line_cur.substr(last_word_pos));
        }

        // ghetto scroll
        while(text_lines.size() > max_line_num) text_lines.pop_front();

        // rebuild text
        text_full.clear();
        for(auto&& text_line : text_lines) {
            text_full += "\n" + text_line;
        }


        // feed sampled char back into model
        run_model(ch, t_state);
    }



    //--------------------------------------------------------------
    void draw() {
        stringstream str;
        str << ofGetFrameRate() << endl;
        str << endl;
        str << "DEL   : clear text " << endl;
        str << "TAB   : auto run (" << do_auto_run << ")" << endl;
        str << "RIGHT : run one char " << endl;
        str << endl;

        str << "Press number key to load model: " << endl;
        for(int i=0; i<model_names.size(); i++) {
            auto marker = (i==cur_model_index) ? ">" : " ";
            str << " " << (i+1) << " : " << marker << " " << model_names[i] << endl;
        }

        str << endl;
        str << "Any other key to type," << endl;
        str << "(and prime the model accordingly)" << endl;
        str << endl;


        if(session) {
            // sample character from probability distribution
            int cur_char_index = msa::tf::sample_from_prob(rng, probs);
            char cur_char = int_to_char[cur_char_index];

            str << "Next char : " << cur_char_index << " | " << cur_char << endl;


            if(do_auto_run || do_run_once) {
                if(do_run_once) do_run_once = false;

                add_char(cur_char);
            }
        }

        // display probability histogram
        msa::tf::draw_probs(probs, ofRectangle(0, 0, ofGetWidth(), ofGetHeight()));


        // draw texts
        ofSetColor(150);
        ofDrawBitmapString(str.str(), 20, 20);

        ofSetColor(0, 200, 0);
        ofDrawBitmapString(text_full + "_", 320, 10);
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
            text_lines = { "The" };
            break;

        case OF_KEY_TAB:
            do_auto_run ^= true;
            break;

        case OF_KEY_RIGHT:
            do_run_once = true;
            do_auto_run = false;
            break;

        default:
            do_auto_run = false;
            if(char_to_int.count(key) > 0) add_char(key);
            break;
        }

    }


};

//========================================================================
int main() {
    ofSetupOpenGL(1280, 720, OF_WINDOW);
    ofRunApp(new ofApp());
}
