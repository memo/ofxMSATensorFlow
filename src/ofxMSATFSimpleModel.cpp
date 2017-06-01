#include "ofxMSATFSimpleModel.h"

namespace msa {
namespace tf {


//--------------------------------------------------------------
SimpleModel::SimpleModel(string model_path,
                         vector<string> input_op_names,
                         vector<string> output_op_names,
                         string name,
                         const string device,
                         const Session_ptr session,
                         const tensorflow::SessionOptions& session_options) {
    setup(model_path, input_op_names, output_op_names, name, device, session, session_options);
}


//--------------------------------------------------------------
void SimpleModel::setup(string model_path,
                        vector<string> input_op_names,
                        vector<string> output_op_names,
                        string name,
                        const string device,
                        const Session_ptr session,
                        const tensorflow::SessionOptions& session_options) {
    ofLogVerbose("msa::tf::SimpleModel")
            << "SimpleModel " << name
            << " model_path: " << model_path;
//            << " input_op_name: " << input_op_names
//            << " output_op_name: " << output_op_names
//            << " device: " << device;

    close();

    this->model_path = model_path;
    this->input_op_names = input_op_names;
    this->output_op_names = output_op_names;
    this->name = (name=="") ? model_path : name;
    this->session = session;

    this->graph_def = load_graph_def(model_path);
    if(!graph_def) {
        ofLogError("msa::tf::SimpleModel") << missing_data_error();
        return;
    }

    if(!this->session) {
        // if we don't have a session yet, create one and load graph def into it
        this->session = create_session_with_graph(this->graph_def, device, session_options);
    } else {
        // if we do have a session, load graph def into it
        create_graph_in_session(this->session, this->graph_def, device);
    }

    // prepare input tensors
    // ideally read tensor type & shape from the graph_def and allocate tensors correctly
    // for now init empty tensors, and user should call init_inputs()
    for(const auto& op_name : input_op_names) this->input_tensors.push_back(make_pair(op_name, tensorflow::Tensor()));
}


//--------------------------------------------------------------
void SimpleModel::init_inputs(tensorflow::DataType type, const vector<tensorflow::int64>& shape, int tensor_index) {
//void SimpleModel::init_inputs(tensorflow::DataType type, const tensorflow::TensorShape& shape, int tensor_index) {
    this->input_tensors[tensor_index].second = tensorflow::Tensor(type, tensorflow::TensorShape(shape));
}


//--------------------------------------------------------------
bool SimpleModel::run() {
    if(!this->is_loaded()) {
        ofLogWarning("msa::tf::SimpleModel") << "Trying to run " << name << " when not loaded";
        return false;
    }


    // run graph, feed input tensors, fetch output tensors
    tensorflow::Status status = session->Run(this->input_tensors, this->output_op_names, {}, &this->output_tensors);
    log_error(status, "msa::tf::SimpleModel");
    return status.ok();
}


//--------------------------------------------------------------
void SimpleModel::close() {
    input_op_names.clear();
    output_op_names.clear();

    session = nullptr;
    graph_def = nullptr;

    input_tensors.clear();
    output_tensors.clear();
}



}
}
