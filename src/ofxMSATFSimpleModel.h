/*
 * a simple wrapper for a basic model
 * variable number of input nodes (tensors of variable rank and dimensions)
 * variable number of output nodes (tensors of variable rank and dimensions)
 *
 * for a simple barebones example see example-pix2pix-simple
 *
 * for information on how to prepare models for ofxMSATensorflow see
 * https://github.com/memo/ofxMSATensorFlow/wiki/Preparing-models-for-ofxMSATensorFlow
 */

#pragma once

#include "ofxMSATFIncludes.h"
#include "ofxMSATFUtils.h"

namespace msa {
namespace tf {

class SimpleModel
{
public:
    typedef shared_ptr<SimpleModel> Ptr;

    //--------------------------------------------------------------
    // empty constructor does nothing, call setup later.
    SimpleModel() {}

    //--------------------------------------------------------------
    // pass everything in constructor (e.g. if using as Ptr)
    // if a session is passed in, uses that, otherwise creates a new session using session_options
    // see ofxMSATFUtils for simple way of constructing common session_options
    SimpleModel(string model_path, // path to model (graph definition), assuming frozen .pb protobuf
                vector<string> input_op_names, // list of tensor names to feed to (i.e. inputs to the model)
                vector<string> output_op_names, // list of tensor names to fetch (i.e. outputs of the model)
                string name="", // arbitrary human readable name for this model
                const string device="",   // "/cpu:0", "/gpu:0" etc.
                const Session_ptr session=nullptr, // if null, SimpleModel initialises a new session, otherwise uses this
                const tensorflow::SessionOptions& session_options=tensorflow::SessionOptions());


    //--------------------------------------------------------------
    // alternatively, call setup on an existing SimpleModel
    void setup(string model_path,
               vector<string> input_op_names,
               vector<string> output_op_names,
               string name="",
               const string device="",
               const Session_ptr session=nullptr,
               const tensorflow::SessionOptions& session_options=tensorflow::SessionOptions());



    //--------------------------------------------------------------
    // IMPORTANT: then initialise input tensors to specified type and shape
    // tensor_index is which input tensor to init (if there is more than one). order is same as input_op_names
    // (TODO: ideally the SimpleModel constructor or setup would read this info from the graph_def and call this internally)
    void init_inputs(tensorflow::DataType type, const vector<tensorflow::int64>& shape, int tensor_index=0);
    //    void init_inputs(tensorflow::DataType type, const tensorflow::TensorShape& shape, int tensor_index=0);


    //--------------------------------------------------------------
    // getters
    bool is_loaded() const                                      { return session != nullptr; }

    string get_name() const                                     { return name; }
    string get_model_path() const                               { return model_path; }

    const vector<string>& get_input_op_names() const            { return input_op_names; }
    const vector<string>& get_output_op_names() const           { return output_op_names; }

    Session_ptr& get_session()                                  { return session; }
    const Session_ptr& get_session() const                      { return session; }

    GraphDef_ptr& get_graph_def()                               { return graph_def; }
    const GraphDef_ptr& get_graph_def() const                   { return graph_def; }

    // access input tensor by index (these are stored in a weird format, see below)
    // if model has only one input node, you can omit i (i.e. i=0)
    tensorflow::Tensor& get_input_tensor(int i=0)               { return input_tensors[i].second; }
    const tensorflow::Tensor& get_input_tensor(int i=0) const   { return input_tensors[i].second; }

    // access output tensor by index
    // if model has only one output node, you can omit i (i.e. i=0)
    tensorflow::Tensor& get_output_tensor(int i=0)              { return output_tensors[i]; }
    const tensorflow::Tensor& get_output_tensors(int i=0) const { return output_tensors[i]; }


    //--------------------------------------------------------------
    // general purpose 'run' operates on internal tensorflow::tensors
    // hence can work with multiple input tensors and multiple output tensors
    // prior to running this, make sure to write your input data to this->input_tensors
    // output is written to this->output_tensors which is where you can read the results from
    // use tensor <--> OF Format conversion functions in ofxMSATFUtils (to convert between ofImage, ofPixels, std::vector <--> tensorflow::tensor)
    // returns true if successful, otherwise returns false
    bool run();


    //--------------------------------------------------------------
    // convenience methods for common simple cases

    // if the model works with images, conversion between image and tensor done internally
    // when inputting an image:
    //    img_in must be same format (e.g. float32, int etc.) as tensor
    // when outputting an image:
    //    img_out will contain the resulting image. It doesn't have to be pre-allocated, but obviously it will be quicker if it is pre-allocated
    //    output is also stored in this->output_tensors in raw format as usual
    // optional xxx_range parameters are for automatic mapping of values, e.g. 0...1 <--> -1...1 (leave blank to bypass)
    //     i.e. image_range -> model_input_range before going in. model_output_range -> image_range after coming out
    // TODO: assuming batch size 1 for now
    // for more than one image in and out, see the code below and use the general purpose run() method

    template<typename T>
    bool run_image_to_vector(const ofImage_<T>& img_in, vector<T>& vec_out, ofVec2f model_in_range=ofVec2f(), ofVec2f image_range=ofVec2f(0, 1));

    template<typename T>
    bool run_vector_to_image(const vector<T>& vec_in, ofImage_<T>& img_out, ofVec2f model_out_range=ofVec2f(), ofVec2f image_range=ofVec2f(0, 1));

    template<typename T>
    bool run_image_to_image(const ofImage_<T>& img_in, ofImage_<T>& img_out, ofVec2f model_in_range=ofVec2f(), ofVec2f model_out_range=ofVec2f(), ofVec2f image_range=ofVec2f(0, 1));


protected:
    string name;  // human readable name of model (e.g. for a gui)
    string model_path; // path to file containing data (often a frozen graph definition file, .pb)
    vector<string> input_op_names;  // name(s) of operators for input (i.e. to feed)
    vector<string> output_op_names; // name(s) of operators for output (i.e. to fetch)

    msa::tf::Session_ptr session;
    msa::tf::GraphDef_ptr graph_def;

    vector<pair<string, tensorflow::Tensor> > input_tensors; // input(s) to the model (using tensorflow format vector< pair<name, tensor> >)
    vector<tensorflow::Tensor> output_tensors;      // output(s) of the model

    void close();
};


//--------------------------------------------------------------
template<typename T>
bool SimpleModel::run_image_to_vector(const ofImage_<T>& img_in, vector<T>& vec_out, ofVec2f model_in_range, ofVec2f image_range) {
    // copy img_in into input tensor. do not use memcpy. map range as nessecary
    msa::tf::image_to_tensor(img_in, this->get_input_tensor(), false, image_range, model_in_range);

    // run on internal input tensor(s)
    if( this->run() ) {

        // copy output tensor into vec_out. do not use memcpy. no range mapping
        msa::tf::tensor_to_vector(this->get_output_tensor(), vec_out, false);
        return true;
    }
    return false;
}

//--------------------------------------------------------------
template<typename T>
bool SimpleModel::run_vector_to_image(const vector<T>& vec_in, ofImage_<T>& img_out, ofVec2f model_out_range, ofVec2f image_range) {
    // copy vec_in into input tensor. do not use memcpy. no range mapping
    msa::tf::vector_to_tensor(vec_in, this->get_input_tensor(), false);

    // run on internal input tensor(s)
    if( this->run() ) {

        // copy output tensor into img_out. do not use memcpy, map range as nessecary
        msa::tf::tensor_to_image(this->get_output_tensor(), img_out, false, model_out_range, image_range);
        return true;
    }
    return false;
}

//--------------------------------------------------------------
template<typename T>
bool SimpleModel::run_image_to_image(const ofImage_<T>& img_in, ofImage_<T>& img_out, ofVec2f model_in_range, ofVec2f model_out_range, ofVec2f image_range) {
    // copy img_in into input tensor. do not use memcpy. map range as nessecary
    msa::tf::image_to_tensor(img_in, this->get_input_tensor(), false, image_range, model_in_range);

    // run on internal input tensor(s)
    if( this->run() ) {

        // copy output tensor into img_out. do not use memcpy, map range as nessecary
        msa::tf::tensor_to_image(this->get_output_tensor(), img_out, false, model_out_range, image_range);
        return true;
    }
    return false;
}

}
}
