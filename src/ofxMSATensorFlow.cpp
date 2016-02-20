//#include "ofxMSATensorFlow.h"


//namespace msa {
//namespace tf {


////--------------------------------------------------------------
//ofxMSATensorFlow::~ofxMSATensorFlow() {
//    closeSession();
//}


////--------------------------------------------------------------
//// initialize session, return true if successfull
//bool ofxMSATensorFlow::setup(const tensorflow::SessionOptions & session_options) {
//    // close existing sessions (if any)
//    closeSession();

//    // create new session
//    status_ = NewSession(session_options, &session_);

//    // check error and return true if status ok
//    logError();
//    return status_.ok();
//}


////--------------------------------------------------------------
//// load a binary graph file and add to sessoin, return true if successfull
//bool ofxMSATensorFlow::loadGraph(const string path, tensorflow::Env* env, const string device) {
//    // load the file
//    status_ = tensorflow::ReadBinaryProto(env, ofToDataPath(path), &graph_def_);

//    // check status
//    logError();
//    if(!status_.ok()) return false;

//    // add graph to session
//    return createSessionWithGraph(device);
//}


////--------------------------------------------------------------
//// create session with graph
//// called automatically by loadGraph, but call manually if you want to modify the graph
//bool ofxMSATensorFlow::createSessionWithGraph(const string device) {
//    tensorflow::graph::SetDefaultDevice(device, &graph_def_);

//    // add graph to session
//    status_ = session_->Create(graph_def_);

//    // check error and return true if status ok
//    logError();
//    if(status_.ok()) initialized_ = true;
//    return status_.ok();
//}



////--------------------------------------------------------------
//// run the graph with given inputs and outputs, return true if successfull
//// (more: https://www.tensorflow.org/versions/master/api_docs/cc/ClassSession.html#virtual_Status_tensorflow_Session_Run )
//bool ofxMSATensorFlow::run(const std::vector<std::pair<string, tensorflow::Tensor> >& inputs,
//                           const std::vector<string>& output_tensor_names,
//                           const std::vector<string>& target_node_names,
//                           std::vector<tensorflow::Tensor>* outputs) {
//    if(initialized_) {
//        session_->Run(inputs,output_tensor_names, target_node_names, outputs);
//        // check error and return true if status ok
//        logError();
//        return status_.ok();
//    } else {
//        return false;
//    }
//}



////--------------------------------------------------------------
//void ofxMSATensorFlow::closeSession() {
//    if(session_) {
//        session_->Close();
//        session_ = NULL;
//    }
//}



////--------------------------------------------------------------
//void ofxMSATensorFlow::logError() {
//    if(!status_.ok())  {
//        ofLogError() << status_.ToString(); }
//}


//}   // namespace tf
//}   // namespace msa
