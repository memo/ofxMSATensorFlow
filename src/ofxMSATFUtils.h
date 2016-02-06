/*
 * General helper functions
 */

#pragma once

#include "ofxMSATFIncludes.h"

namespace msa {
namespace tf {

// HELPER FUNCTIONS

// return dimensions of an image to hold the Tensor: [x: width, y: height, z: depth (number of channels)]
// use chmap to map channels (1st char: dim_index for x, 2nd char: dim_index for y, 3rd char: dim_index for numchannels)
// default is: if rank==1 -> (w:d0, h:1px, nch:1) | if rank==2 -> (w:d0, h:d1, nch:1) | if rank=>3 -> (w:d1, h:d2, nch:d0)
//      where rank:=number of tensor dimensions (1,2,3 etc.); d0, d1, d2: dim_size in corresponding dimension)
// looks complicated, but it's not (we usually think of images as { width, height, depth },
//      but actually in memory it's { depth, width, height } (if channels are interleaved)
inline ofVec3f tensorToPixelDims(const tensorflow::Tensor &t, string chmap = "120");

//--------------------------------------------------------------
// TENSOR DATA COPY FUNCTIONS
// src & dest must be of same TYPE otherwise will fail/assert/crash! (e.g. Tensor<float>, ofImage<float>, vector<float>
// src & dest can be different shapes but must be SAME NUMBER OF ELEMENTS, bounds checking is not done, will crash if src is larger than dest
// do_memcpy is very fast, but could be dangerous due to alignment issues?
// (PS using explicit function names, because I find it more readable and less bug prone)

// flatten tensor and copy into 1D std::vector. vector will be allocated if nessecary
template<typename T> void tensorToVector(const tensorflow::Tensor &src, std::vector<T> &dst, bool do_memcpy=false);

// copy into pixels. dst won't be reshaped if it's already allocated. otherwise it'll be allocated according to tensorPixelDims(, chmap)
template<typename T> void tensorToPixels(const tensorflow::Tensor &src, ofPixels_<T> &dst, bool do_memcpy=false, string chmap = "120");

// copy into image. dst won't be reshaped if it's already allocated. otherwise it'll be allocated according to tensorPixelDims(, chmap)
template<typename T> void tensorToImage(const tensorflow::Tensor &src, ofImage_<T> &dst, bool do_memcpy=false, string chmap = "120");

// flatten tensor and copy into array. array must already be allocated
template<typename T> void tensorToArray(const tensorflow::Tensor &src, T *dst, bool do_memcpy=false);


// copy std::vector into tensor. tensor must already be allocated and will not be reshaped
template<typename T> void vectorToTensor(const std::vector<T> &src, tensorflow::Tensor &dst, bool do_memcpy=false);

// copy pixels into tensor. tensor must already be allocated and will not be reshaped
template<typename T> void pixelsToTensor(const ofPixels_<T> &src, tensorflow::Tensor &dst, bool do_memcpy=false);

// copy image into tensor. tensor must already be allocated and will not be reshaped
template<typename T> void imageToTensor(const ofImage_<T> &src, tensorflow::Tensor &dst, bool do_memcpy=false);

// copy array into tensor. tensor must already be allocated and will not be reshaped
template<typename T> void arrayToTensor(const T *src, tensorflow::Tensor &dst, bool do_memcpy=false);


//--------------------------------------------------------------
// convert grayscale float image into RGB float image where R -ve and B is +ve
// dst image is allocated if nessecary
template<typename T> void grayToColor(const ofPixels_<T> &src, ofPixels_<T> &dst, float scaler=1.0f);
template<typename T> void grayToColor(const ofImage_<T> &src, ofImage_<T> &dst, float scaler=1.0f);


//--------------------------------------------------------------
// pass in tensor (usually containing scores or probabilities of labels) and number of top items desired
// function returns top_k scores and corresponding indices
inline void getTopScores(tensorflow::Tensor scores_tensor, int topk_count, vector<int> &out_indices, vector<float> &out_scores);


// Takes a file name, and loads a list of labels from it, one per line, and
// returns a vector of the strings. It pads with empty strings so the length
// of the result is a multiple of 16, because our model expects that.
bool readLabelsFile(string file_name, vector<string>& result);


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
ofVec3f tensorToPixelDims(const tensorflow::Tensor &t, string chmap) {
    int rank = t.shape().dims();
    vector<int> tensor_dims(rank);
    for(int i=0; i<rank; i++) tensor_dims[i] = t.dim_size(i); // useful for debugging

    // add z to end of string to top it up to length 3, this'll make it default to 1
    while(chmap.length()<3) chmap += "z";

    // which tensor dimension to use for which image xyz component
    // initially read from chmap parameter
    ofVec3f dim_indices(chmap[0]-'0', chmap[1]-'0', chmap[2]-'0');

    // if tensor rank is less than the chmap, adjust dim_indices accordingly (
    if(rank < chmap.length()) {
        if(rank == 1) {
            //  if(dim_indices)
            dim_indices.set(0, 99, 99);   // set these large so they default to 1
        } else if(rank == 2) {
            if(dim_indices[1] > dim_indices[0]) dim_indices.set(0, 1, 99);
            else dim_indices.set(1, 0, 99);
        }
    }

    ofVec3f image_dims (
                (rank > dim_indices[0] ? t.dim_size( dim_indices[0]) : 1),
        (rank > dim_indices[1] ? t.dim_size( dim_indices[1]) : 1),
        (rank > dim_indices[2] ? t.dim_size( dim_indices[2]) : 1)
      );
    return image_dims;
}


//--------------------------------------------------------------
template<typename T> void tensorToVector(const tensorflow::Tensor &src, std::vector<T> &dst, bool do_memcpy) {
    if(dst.size() != src.NumElements()) dst.resize(src.NumElements());
    tensorToArray(src, dst.data(), do_memcpy);
}


//--------------------------------------------------------------
template<typename T> void tensorToPixels(const tensorflow::Tensor &src, ofPixels_<T> &dst, bool do_memcpy, string chmap) {
    if(!dst.isAllocated()) {
        ofVec3f dims(tensorToPixelDims(src, chmap));
        dst.allocate((int)dims.x, (int)dims.y, (int)dims.z);
    }

    tensorToArray(src, dst.getData(), do_memcpy);
}


//--------------------------------------------------------------
template<typename T> void tensorToImage(const tensorflow::Tensor &src, ofImage_<T> &dst, bool do_memcpy, string chmap) {
    if(!dst.isAllocated()) {
        ofVec3f dims(tensorToPixelDims(src, chmap));
        dst.allocate((int)dims.x, (int)dims.y, dims.z == 1 ? OF_IMAGE_GRAYSCALE : (dims.z == 3 ? OF_IMAGE_COLOR : OF_IMAGE_COLOR_ALPHA));
    }
    tensorToPixels(src, dst.getPixels(), do_memcpy);
    dst.update();
}


//--------------------------------------------------------------
template<typename T> void tensorToArray(const tensorflow::Tensor &src, T *dst, bool do_memcpy) {
    auto src_data = src.flat<T>().data();
    int n = src.NumElements();
    if(do_memcpy) memcpy(dst, src_data, n * sizeof(T));
    else for(int i=0; i<n; i++) dst[i] = src_data[i];
}


//--------------------------------------------------------------
template<typename T> void vectorToTensor(const std::vector<T> &src, tensorflow::Tensor &dst, bool do_memcpy) {
    arrayToTensor(src.data(), dst, do_memcpy);
}


//--------------------------------------------------------------
template<typename T> void pixelsToTensor(const ofPixels_<T> &src, tensorflow::Tensor &dst, bool do_memcpy) {
    arrayToTensor(src.getData(), dst, do_memcpy);
}


//--------------------------------------------------------------
template<typename T> void imageToTensor(const ofImage_<T> &src, tensorflow::Tensor &dst, bool do_memcpy) {
    pixelsToTensor(src.getPixels(), dst, do_memcpy);
}


//--------------------------------------------------------------
template<typename T> void arrayToTensor(const T *in, tensorflow::Tensor &dst, bool do_memcpy) {
    auto dst_data = dst.flat<T>().data();
    int n = dst.NumElements();
    if(do_memcpy) memcpy(dst_data, in, n * sizeof(T));
    else for(int i=0; i<n; i++) dst_data[i] = in[i];
}


//--------------------------------------------------------------
template<typename T> void grayToColor(const ofPixels_<T> &src, ofPixels_<T> &dst, float scaler) {
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
template<typename T> void grayToColor(const ofImage_<T> &src, ofImage_<T> &dst, float scaler) {
    grayToColor(src.getPixels(), dst.getPixels(), scaler);
    dst.update();
}


//--------------------------------------------------------------
void getTopScores(tensorflow::Tensor scores_tensor, int topk_count, vector<int> &out_indices, vector<float> &out_scores) {
    tensorflow::GraphDefBuilder b;
    string output_name = "top_k";
    tensorflow::ops::TopK(tensorflow::ops::Const(scores_tensor, b.opts()), topk_count, b.opts().WithName(output_name));

    // This runs the GraphDef network definition that we've just constructed, and
    // returns the results in the output tensors.
    tensorflow::GraphDef graph;
    b.ToGraphDef(&graph);

    std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
    session->Create(graph);

    // The TopK node returns two outputs, the scores and their original indices,
    // so we have to append :0 and :1 to specify them both.
    std::vector<tensorflow::Tensor> output_tensors;
    session->Run({}, {output_name + ":0", output_name + ":1"},{}, &output_tensors);
    tensorToVector(output_tensors[0], out_scores);
    tensorToVector(output_tensors[1], out_indices);
}


//--------------------------------------------------------------
bool readLabelsFile(string file_name, vector<string>& result) {
    std::ifstream file(file_name);
    if (!file) {
        ofLogError() <<"ReadLabelsFile: " << file_name << " not found.";
        return false;
    }

    result.clear();
    string line;
    while (std::getline(file, line)) {
        result.push_back(line);
    }
    const int padding = 16;
    while (result.size() % padding) {
        result.emplace_back();
    }
    return true;
}


}   // namespace tf
}   // namespace msa
