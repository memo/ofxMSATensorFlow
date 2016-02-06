#include "ofxMSATensorFlow.h"


namespace msa {
namespace tf {


//--------------------------------------------------------------
ofxMSATensorFlow::~ofxMSATensorFlow() {
    closeSession();
}


//--------------------------------------------------------------
// initialize session, return true if successfull
bool ofxMSATensorFlow::setup(const tensorflow::SessionOptions & session_options) {
    // close existing sessions (if any)
    closeSession();

    // create new session
    _status = NewSession(session_options, &_session);

    // check error and return true if status ok
    logError();
    return _status.ok();
}


//--------------------------------------------------------------
// load a binary graph file and add to sessoin, return true if successfull
bool ofxMSATensorFlow::loadGraph(const string path, tensorflow::Env* env) {
    // load the file
    _status = tensorflow::ReadBinaryProto(env, ofToDataPath(path), &_graph_def);

    // check status
    logError();
    if(!_status.ok()) return false;

    // add graph to session
    return createSessionWithGraph();
}


//--------------------------------------------------------------
// create session with graph
// called automatically by loadGraph, but call manually if you want to modify the graph
bool ofxMSATensorFlow::createSessionWithGraph() {
    /* if (options.target.empty()) {
        graph::SetDefaultDevice(opts->use_gpu ? "/gpu:0" : "/cpu:0", &def);
    }
*/
    // add graph to session
    _status = _session->Create(_graph_def);

    // check error and return true if status ok
    logError();
    if(_status.ok()) _initialized = true;
    return _status.ok();
}



//--------------------------------------------------------------
// run the graph with given inputs and outputs, return true if successfull
// (more: https://www.tensorflow.org/versions/master/api_docs/cc/ClassSession.html#virtual_Status_tensorflow_Session_Run )
bool ofxMSATensorFlow::run(const std::vector<std::pair<string, tensorflow::Tensor> >& inputs,
                           const std::vector<string>& output_tensor_names,
                           const std::vector<string>& target_node_names,
                           std::vector<tensorflow::Tensor>* outputs) {
    if(_initialized) {
        _session->Run(inputs,output_tensor_names, target_node_names, outputs);
        // check error and return true if status ok
        logError();
        return _status.ok();
    } else {
        return false;
    }
}



//--------------------------------------------------------------
void ofxMSATensorFlow::closeSession() {
    if(_session) {
        _session->Close();
        _session = NULL;
    }
}



//--------------------------------------------------------------
void ofxMSATensorFlow::logError() {
    if(!_status.ok())  {
        ofLogError() << _status.ToString(); }
}


}   // namespace tf
}   // namespace msa
