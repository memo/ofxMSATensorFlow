#include "ofxMSATFSimpleModel.h"

namespace msa {
namespace tf {


//--------------------------------------------------------------
SimpleModel::SimpleModel(string model_path,
                                 vector<string> input_op_names,
                                 vector<string> output_op_names,
                                 string name,
                                 const string device,
                                 const tensorflow::SessionOptions& session_options) {
    setup(model_path, input_op_names, output_op_names, name, device, session_options);
}


//--------------------------------------------------------------
void SimpleModel::setup(string model_path,
                                 vector<string> input_op_names,
                                 vector<string> output_op_names,
                                 string name,
                                 const string device,
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

    this->graph_def = load_graph_def(model_path);
    this->session = create_session_with_graph(this->graph_def, device, session_options);

    // prepare input tensors
    // ideally read tensor type & shape from the graph_def and allocate tensors correctly
    // for now init empty tensors, and user should call init_inputs()
    for(const auto& op_name : input_op_names) this->input_tensors.push_back(make_pair(op_name, tensorflow::Tensor()));
}


//--------------------------------------------------------------
void SimpleModel::init_inputs(tensorflow::DataType type, const tensorflow::TensorShape& shape, int tensor_index) {
    this->input_tensors[tensor_index].second = tensorflow::Tensor(type, shape);
}


//--------------------------------------------------------------
bool SimpleModel::run() {
    if(!this->is_loaded()) {
        ofLogWarning("msa::tf::SimpleModel") << "Trying to run " << name << " when not loaded";
        return false;
    }


    // run graph, feed input tensors, fetch output tensors
    tensorflow::Status status = session->Run(this->input_tensors, this->output_op_names, {}, &this->output_tensors);
    if(status != tensorflow::Status::OK()) {
        ofLogError("msa::tf::SimpleModel") << status.error_message();
        return false;
    }

    return true;
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
