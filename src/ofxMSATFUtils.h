/*
 * General helper functions
 */

#pragma once

#include "ofxMSATFIncludes.h"

namespace msa {
namespace tf {

// some typedefs to shared pointers
typedef shared_ptr<tensorflow::Session> Session_ptr;
typedef shared_ptr<tensorflow::GraphDef> GraphDef_ptr;



// COMMON LOW LEVEL FUNCTIONAL STUFF (i.e. stateless).
// TODO update these to r1.0, Using tensorflow::Scope


// check status for error and log if error found. return status as is
tensorflow::Status log_error(const tensorflow::Status& status, const string msg);


// load a graph definition from file, return as shared pointer
GraphDef_ptr load_graph_def(const string path, tensorflow::Env* env = tensorflow::Env::Default());


// create and initialize session with graph definition, return shared pointer to session
Session_ptr create_session_with_graph(
        tensorflow::GraphDef& graph_def,
        const string device = "",   // "/cpu:0", "/gpu:0" etc.
        const tensorflow::SessionOptions& session_options = tensorflow::SessionOptions());


// convenience method, same as above, but takes pointer
Session_ptr create_session_with_graph(
        GraphDef_ptr pgraph_def,
        const string device = "",   // "/cpu:0", "/gpu:0" etc.
        const tensorflow::SessionOptions& session_options = tensorflow::SessionOptions());


// convenience method, same as above, but takes graph filename
Session_ptr create_session_with_graph(
        const string graph_def_path,
        const string device = "",   // "/cpu:0", "/gpu:0" etc.
        const tensorflow::SessionOptions& session_options = tensorflow::SessionOptions());


// pass in tensor (e.g. of probabilities) and number of top items desired, returns top k values and corresponding indices
//void get_top_scores(tensorflow::Tensor scores_tensor, int topk_count, vector<int> &out_indices, vector<float> &out_scores, string output_name = "top_k");

// pass in vector (e.g. of probabilities) and number of top items desired, returns top k values and corresponding indices
void get_topk(const vector<float> probs, vector<int> &out_indices, vector<float> &out_values, int k=10);


// Takes a file name, and loads a list of labels from it, one per line, and
// returns a vector of the strings. It pads with empty strings so the length
// of the result is a multiple of 16, because our model expects that.
bool read_labels_file(string file_name, vector<string>& result);



//--------------------------------------------------------------
//--------------------------------------------------------------
//--------------------------------------------------------------
// HELPER CONVERSION FUNCTIONS

// return dimensions of an image to hold the Tensor: [x: width, y: height, z: depth (number of channels)]
// use chmap to map channels (1st char: dim_index for x, 2nd char: dim_index for y, 3rd char: dim_index for numchannels)
// default is: if rank==1 -> (w:d0, h:1px, nch:1) | if rank==2 -> (w:d0, h:d1, nch:1) | if rank=>3 -> (w:d1, h:d2, nch:d0)
//      where rank:=number of tensor dimensions (1,2,3 etc.); d0, d1, d2: dim_size in corresponding dimension)
// looks complicated, but it's not (we usually think of images as { width, height, depth },
//      but actually in memory it's { height, width, depth } (if channels are interleaved)
vector<int> tensor_to_pixel_dims(const tensorflow::Tensor &t, string chmap = "102");

//--------------------------------------------------------------
// TENSOR DATA COPY FUNCTIONS
// src & dest must be of same TYPE otherwise will fail/assert/crash! (e.g. Tensor<float>, ofImage<float>, vector<float>
// src & dest can be different shapes but must be SAME NUMBER OF ELEMENTS, bounds checking is not done, will crash if src is larger than dest
// do_memcpy is very fast, but could be dangerous due to alignment issues?
// (PS using explicit function names, because I find it more readable and less bug prone)

// flatten tensor and copy into 1D std::vector. vector will be allocated if nessecary
template<typename T> void tensor_to_vector(const tensorflow::Tensor &src, std::vector<T> &dst, bool do_memcpy=false);

// copy into pixels. dst won't be reshaped if it's already allocated. otherwise it'll be allocated according to tensorPixelDims(, chmap)
template<typename T> void tensor_to_pixels(const tensorflow::Tensor &src, ofPixels_<T> &dst, bool do_memcpy=false, string chmap = "102");

// copy into image. dst won't be reshaped if it's already allocated. otherwise it'll be allocated according to tensorPixelDims(, chmap)
template<typename T> void tensor_to_image(const tensorflow::Tensor &src, ofImage_<T> &dst, bool do_memcpy=false, string chmap = "102");

// flatten tensor and copy into array. array must already be allocated
template<typename T> void tensor_to_array(const tensorflow::Tensor &src, T *dst, bool do_memcpy=false);


// copy std::vector into tensor. tensor must already be allocated and will not be reshaped
template<typename T> void vector_to_tensor(const std::vector<T> &src, tensorflow::Tensor &dst, bool do_memcpy=false);

// copy pixels into tensor. tensor must already be allocated and will not be reshaped
template<typename T> void pixels_to_tensor(const ofPixels_<T> &src, tensorflow::Tensor &dst, bool do_memcpy=false);

// copy image into tensor. tensor must already be allocated and will not be reshaped
template<typename T> void image_to_tensor(const ofImage_<T> &src, tensorflow::Tensor &dst, bool do_memcpy=false);

// copy array into tensor. tensor must already be allocated and will not be reshaped
template<typename T> void array_to_tensor(const T *src, tensorflow::Tensor &dst, bool do_memcpy=false);

// copy array into tensor. tensor must already be allocated and will not be reshaped
template<typename T> void scalar_to_tensor(const T src, tensorflow::Tensor &dst);


//--------------------------------------------------------------
// TENSOR DATA CONVERSION FUNCTIONS
// different to the above, these alloc and return the target type (instead of writing to a parameter)

// flatten tensor and return std::vector
template<typename T> std::vector<T> tensor_to_vector(const tensorflow::Tensor &src, bool do_memcpy=false);

// return pixels. it'll be allocated according to tensorPixelDims(, chmap)
template<typename T> ofPixels_<T> tensor_to_pixels(const tensorflow::Tensor &src, bool do_memcpy=false, string chmap = "102");

// return image. it'll be allocated according to tensorPixelDims(, chmap)
template<typename T> ofImage_<T> tensor_to_image(const tensorflow::Tensor &src, bool do_memcpy=false, string chmap = "102");

// return scalar value
template<typename T> T tensor_to_scalar(const tensorflow::Tensor &src);


// return tensor from vector
template<typename T> tensorflow::Tensor vector_to_tensor(const std::vector<T> &src, bool do_memcpy=false);

// return tensor from pixels
template<typename T> tensorflow::Tensor pixels_to_tensor(const ofPixels_<T> &src, bool do_memcpy=false);//, string chmap = "102");

// return tensor from image
template<typename T> tensorflow::Tensor image_to_tensor(const ofImage_<T> &src, bool do_memcpy=false);//, string chmap = "102");

// return tensor from scalar
template<typename T> tensorflow::Tensor scalar_to_tensor(const T src);


//--------------------------------------------------------------
// convert grayscale float image into RGB float image where R -ve and B is +ve
// dst image is allocated if nessecary
template<typename T> void gray_to_color(const ofPixels_<T> &src, ofPixels_<T> &dst, float scaler=1.0f);
template<typename T> void gray_to_color(const ofImage_<T> &src, ofImage_<T> &dst, float scaler=1.0f);


// IMPLEMENTATIONS

//--------------------------------------------------------------
//--------------------------------------------------------------
//--------------------------------------------------------------

//--------------------------------------------------------------
//--------------------------------------------------------------
//--------------------------------------------------------------

//--------------------------------------------------------------
//--------------------------------------------------------------
//--------------------------------------------------------------

//--------------------------------------------------------------
//--------------------------------------------------------------
//--------------------------------------------------------------

//--------------------------------------------------------------
template<typename T> void tensor_to_vector(const tensorflow::Tensor &src, std::vector<T> &dst, bool do_memcpy) {
    if(dst.size() != src.NumElements()) dst.resize(src.NumElements());
    tensor_to_array(src, dst.data(), do_memcpy);
}


//--------------------------------------------------------------
template<typename T> void tensor_to_pixels(const tensorflow::Tensor &src, ofPixels_<T> &dst, bool do_memcpy, string chmap) {
    if(!dst.isAllocated()) {
        auto dims(tensor_to_pixel_dims(src, chmap));
        dst.allocate(dims[0], dims[1], dims[2]);
    }

    tensor_to_array(src, dst.getData(), do_memcpy);
}


//--------------------------------------------------------------
template<typename T> void tensor_to_image(const tensorflow::Tensor &src, ofImage_<T> &dst, bool do_memcpy, string chmap) {
    if(!dst.isAllocated()) {
        auto dims(tensor_to_pixel_dims(src, chmap));
        dst.allocate(dims[0], dims[1], dims[2] == 1 ? OF_IMAGE_GRAYSCALE : (dims[2] == 3 ? OF_IMAGE_COLOR : OF_IMAGE_COLOR_ALPHA));
    }
    tensor_to_pixels(src, dst.getPixels(), do_memcpy);
    dst.update();
}


//--------------------------------------------------------------
template<typename T> void tensor_to_array(const tensorflow::Tensor &src, T *dst, bool do_memcpy) {
    auto src_data = src.flat<T>().data();
    int n = src.NumElements();
    if(do_memcpy) memcpy(dst, src_data, n * sizeof(T));
    else for(int i=0; i<n; i++) dst[i] = src_data[i];
}


//--------------------------------------------------------------
template<typename T> void vector_to_tensor(const std::vector<T> &src, tensorflow::Tensor &dst, bool do_memcpy) {
    array_to_tensor(src.data(), dst, do_memcpy);
}


//--------------------------------------------------------------
template<typename T> void pixels_to_tensor(const ofPixels_<T> &src, tensorflow::Tensor &dst, bool do_memcpy) {
    array_to_tensor(src.getData(), dst, do_memcpy);
}


//--------------------------------------------------------------
template<typename T> void image_to_tensor(const ofImage_<T> &src, tensorflow::Tensor &dst, bool do_memcpy) {
    pixels_to_tensor(src.getPixels(), dst, do_memcpy);
}


//--------------------------------------------------------------
template<typename T> void array_to_tensor(const T *in, tensorflow::Tensor &dst, bool do_memcpy) {
    auto dst_data = dst.flat<T>().data();
    int n = dst.NumElements();
    if(do_memcpy) memcpy(dst_data, in, n * sizeof(T));
    else for(int i=0; i<n; i++) dst_data[i] = in[i];
}


//--------------------------------------------------------------
template<typename T> void scalar_to_tensor(const T src, tensorflow::Tensor &dst) {
    dst.scalar<T>()() = src;
}


//--------------------------------------------------------------
//--------------------------------------------------------------
template<typename T> std::vector<T> tensor_to_vector(const tensorflow::Tensor &src, bool do_memcpy) {
    std::vector<T> dst;
    tensor_to_vector(src, dst, do_memcpy);
    return dst;
}


//--------------------------------------------------------------
template<typename T> ofPixels_<T> tensor_to_pixels(const tensorflow::Tensor &src, bool do_memcpy, string chmap){
    ofPixels_<T> dst;
    tensor_to_pixels(src, dst, do_memcpy, chmap);
    return dst;
}


//--------------------------------------------------------------
template<typename T> ofImage_<T> tensor_to_image(const tensorflow::Tensor &src, bool do_memcpy, string chmap) {
    ofImage_<T> dst;
    tensor_to_image(src, dst, do_memcpy, chmap);
    return dst;
}


//--------------------------------------------------------------
template<typename T> T tensor_to_scalar(const tensorflow::Tensor &src) {
    return src.scalar<T>()();
}


//--------------------------------------------------------------
template<typename T> tensorflow::Tensor vector_to_tensor(const std::vector<T> &src, bool do_memcpy) {
    tensorflow::Tensor dst( tensorflow::Tensor(tensorflow::DT_FLOAT, {src.size()}).vec<T>() );   // initting tensor with float, converting later
    vector_to_tensor(src, dst, do_memcpy);
    return dst;
}

//--------------------------------------------------------------
template<typename T> tensorflow::Tensor pixels_to_tensor(const ofPixels_<T> &src, bool do_memcpy) {//, string chmap) {}
   tensorflow::Tensor dst( tensorflow::Tensor(tensorflow::DT_FLOAT, { src.getNumChannels(), src.getWidth(), src.getHeight()}) );
   pixels_to_tensor(src, dst, do_memcpy);
   return dst;
}

//--------------------------------------------------------------
template<typename T> tensorflow::Tensor image_to_tensor(const ofImage_<T> &src, bool do_memcpy) {//, string chmap) {}
    return pixels_to_tensor(src.getPixels(), do_memcpy);
}


//--------------------------------------------------------------
template<typename T> tensorflow::Tensor scalar_to_tensor(const T src) {
    tensorflow::Tensor dst = tensorflow::Tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape());// = tensorflow::Tensor(tensorflow::DT_FLOAT, {1}).scalar<T>();
    scalar_to_tensor(src, dst);
    return dst;
}


//--------------------------------------------------------------
//--------------------------------------------------------------
template<typename T> void gray_to_color(const ofPixels_<T> &src, ofPixels_<T> &dst, float scaler) {
    dst.allocate(src.getWidth(), src.getHeight(), 3);
    const T *src_data = src.getData();
    T *dst_data = dst.getData();
    for(int i=0; i < src.getWidth() * src.getHeight(); i++) {
        float f = src_data[i] * scaler;
        dst_data[i*3] = f < 0 ? -f : 0;
        dst_data[i*3+1] = 0;
        dst_data[i*3+2] = f > 0 ? f : 0;
    }
}

//--------------------------------------------------------------
template<typename T> void gray_to_color(const ofImage_<T> &src, ofImage_<T> &dst, float scaler) {
    gray_to_color(src.getPixels(), dst.getPixels(), scaler);
    dst.update();
}



}   // namespace tf
}   // namespace msa
