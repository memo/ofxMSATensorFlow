/*
 * a simple wrapper for a basic predictor model
 *
 *
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
    // or pass everything in constructor (E.g. if using as Ptr)
    SimpleModel(string model_path,
                    vector<string> input_op_names,
                    vector<string> output_op_names,
                    string name="",
                    const string device="",   // "/cpu:0", "/gpu:0" etc.
                    const tensorflow::SessionOptions& session_options=tensorflow::SessionOptions());


    //--------------------------------------------------------------
    // or call setup
    void setup(string model_path,
               vector<string> input_op_names,
               vector<string> output_op_names,
               string name="",
               const string device="",   // "/cpu:0", "/gpu:0" etc.
               const tensorflow::SessionOptions& session_options=tensorflow::SessionOptions());



    //--------------------------------------------------------------
    // then initialise input tensors to specified type and shape
    // tensor_index is which input tensor to init (if there is more than one). order is same as input_op_names
    // (ideally the SimpleModel constructor or setup would read this info from the graph_def and call this internally)
    void init_inputs(tensorflow::DataType type, const tensorflow::TensorShape& shape, int tensor_index=0);


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

    tensorflow::Tensor& get_input_tensor(int i=0)               { return input_tensors[i].second; } // .first is the name
    const tensorflow::Tensor& get_input_tensor(int i=0) const   { return input_tensors[i].second; }

    tensorflow::Tensor& get_output_tensor(int i=0)              { return output_tensors[i]; }
    const tensorflow::Tensor& get_output_tensors(int i=0) const { return output_tensors[i]; }


    //--------------------------------------------------------------
    // run the model on this->input_tensors
    // output is written to this->output_tensors
    // returns true if successful, otherwise returns false
    // use tensor <--> OF Format conversion functions in ofxMSATFUtils (to convert ofImage, ofPixels, std::vector <--> tensor)
    bool run();


    //--------------------------------------------------------------
    // convenience methods for run

    // if the model expects an image, conversion to tensor done internally
    // output written to this->output_tensors as usual
    // img_in must be same format (e.g. float32, int etc.) as tensor!
    // optional xxx_range parameters are for automatic mapping of values, e.g. 0...1 <--> -1...1 (leave blank to bypass)
    // (image_range -> model_input_range before going in. model_output_range -> image_range after coming out)
    // TODO: assuming batch size 1 for now

    // if model expects an image
    template<typename T>
    bool run(const ofImage_<T>& img_in, ofVec2f model_in_range=ofVec2f(), ofVec2f image_range=ofVec2f(0, 1));

    // if the model also outputs an image, conversion to tensor done internally
    // output image written to img_out (doesn't have to be pre-allocated, but if it is pre-allocated, it will be quicker
    template<typename T>
    bool run(const ofImage_<T>& img_in, ofImage_<T>& img_out, ofVec2f model_in_range=ofVec2f(), ofVec2f model_out_range=ofVec2f(), ofVec2f image_range=ofVec2f(0, 1));


protected:
    string model_path; // path to file containing model data
    vector<string> input_op_names;  // name(s) of operators for input (i.e. to feed)
    vector<string> output_op_names; // name(s) of operators for output (i.e. to fetch)
    string name;  // name of model (e.g. for gui)

    msa::tf::Session_ptr session;
    msa::tf::GraphDef_ptr graph_def;

    vector<pair<string, tensorflow::Tensor> > input_tensors; // input(s) to the model (using tensorflow format vector< pair<name, tensor> >)
    vector<tensorflow::Tensor> output_tensors;      // output(s) of the model

    void close();
};


//--------------------------------------------------------------
template<typename T>
bool SimpleModel::run(const ofImage_<T>& img_in, ofVec2f model_in_range, ofVec2f image_range) {
    // dump img_in into input tensor. do not use memcpy. map range as nessecary
    msa::tf::image_to_tensor(img_in, this->get_input_tensor(), false, image_range, model_in_range);
    return this->run();
}


//--------------------------------------------------------------
template<typename T>
bool SimpleModel::run(const ofImage_<T>& img_in, ofImage_<T>& img_out, ofVec2f model_in_range, ofVec2f model_out_range, ofVec2f image_range) {
    if( this->run(img_in, model_in_range, image_range) ) {
        // dump output tensor into img_out. do not use memcpy, map range as nessecary
        msa::tf::tensor_to_image(this->get_output_tensor(), img_out, false, model_out_range, image_range);
    }
}

}
}
