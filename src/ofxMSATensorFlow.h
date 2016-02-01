/*
 this is simple wrapper for a single session and graph, should suffice for most cases
 you can access the internal variables directly if you need more advanced setup
 I didn't wrap it too much as I think it's important to understand how TensorFlow works,
 in case you need to switch to raw tensorflow project etc.

 There's also a bunch of helper functions for various functions (eg vector <--> tensor <--> image conversions)

*/

#pragma once

#include "ofMain.h"

#ifdef Success
#undef Success  // /usr/include/X11/X.h is defining this as int making compile fail
#endif

#ifdef Status
#undef Status   // /usr/include/X11/Xlib.h is defining this as int making compile fail
#endif

#ifdef None
#undef None   // /usr/include/X11/X.h is defining this as 0 making compile fail
#endif

#ifdef Complex
#undef Complex   // /usr/include/X11/X.h is defining this as 0 making compile fail
#endif

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/tensor.h"
#include "tensorflow/cc/ops/standard_ops.h"

//--------------------------------------------------------------
class ofxMSATensorFlow {
public:

    // HELPER FUNCTIONS

    // return dimensions of an image to hold the Tensor: [x: width, y: height, z: depth (number of channels)]
    // use chmap to map channels (1st char: dim_index for x, 2nd char: dim_index for y, 3rd char: dim_index for numchannels)
    // default is: if rank==1 -> (w:d0, h:1px, nch:1) | if rank==2 -> (w:d0, h:d1, nch:1) | if rank=>3 -> (w:d1, h:d2, nch:d0)
    //      where rank:=number of tensor dimensions (1,2,3 etc.); d0, d1, d2: dim_size in corresponding dimension)
    // looks complicated, but it's not (we usually think of images as { width, height, depth },
    //      but actually in memory it's { depth, width, height } (if channels are interleaved)
    inline static ofVec3f tensorToPixelDims(const tensorflow::Tensor &t, string chmap = "120");

    //--------------------------------------------------------------
    // TENSOR DATA COPY FUNCTIONS
    // src & dest must be of same TYPE otherwise will fail/assert/crash! (e.g. Tensor<float>, ofImage<float, vector<float>
    // src & dest can be different shapes but must be SAME NUMBER OF ELEMENTS, bounds checking is not done, will crash if src is larger than dest
    // do_memcpy is very fast, but could be dangerous due to alignment issues?
    // (PS using explicit function names, because I find it more readable and less bug prone

    // flatten tensor and copy into std::vector. vector will be allocated if nessecary
    template<typename T>
    static void tensorToVector(const tensorflow::Tensor &src, std::vector<T> &dst, bool do_memcpy=false);

    // copy into pixels. dst won't be reshaped if it's already allocated. otherwise it'll be allocated according to tensorPixelDims(, chmap)
    template<typename T>
    static void tensorToPixels(const tensorflow::Tensor &src, ofPixels_<T> &dst, bool do_memcpy=false, string chmap = "120");

    // copy into image. dst won't be reshaped if it's already allocated. otherwise it'll be allocated according to tensorPixelDims(, chmap)
    template<typename T>
    static void tensorToImage(const tensorflow::Tensor &src, ofImage_<T> &dst, bool do_memcpy=false, string chmap = "120");

    // flatten tensor and copy into array. array must already be allocated
    template<typename T>
    static void tensorToArray(const tensorflow::Tensor &src, T *dst, bool do_memcpy=false);


    // copy std::vector into flattened tensor. tensor must already be allocated and will not be reshaped
    template<typename T>
    static void vectorToTensor(const std::vector<T> &src, tensorflow::Tensor &dst, bool do_memcpy=false);

    // copy pixels into tensor. tensor must already be allocated and will not be reshaped
    template<typename T>
    static void pixelsToTensor(const ofPixels_<T> &src, tensorflow::Tensor &dst, bool do_memcpy=false);

    // copy image into tensor. tensor must already be allocated and will not be reshaped
    template<typename T>
    static void imageToTensor(const ofImage_<T> &src, tensorflow::Tensor &dst, bool do_memcpy=false);

    // copy array into flattened tensor. tensor must already be allocated and will not be reshaped
    template<typename T>
    static void arrayToTensor(const T *src, tensorflow::Tensor &dst, bool do_memcpy=false);


    //--------------------------------------------------------------
    // convert grayscale float image into RGB float image where R -ve and B is +ve
    // dst image is allocated if nessecary
    template<typename T>
    static void grayToColor(const ofPixels_<T> &src, ofPixels_<T> &dst, float scaler=1.0f);

    template<typename T>
    static void grayToColor(const ofImage_<T> &src, ofImage_<T> &dst, float scaler=1.0f);



    //--------------------------------------------------------------
    // pass in tensor (usually containing scores or probabilities of labels) and number of top items desired
    // function returns topk scores and corresponding indices
    inline static void getTopScores(tensorflow::Tensor scores_tensor, int topk_count, vector<int> &out_indices, vector<float> &out_scores);


    //--------------------------------------------------------------
    //--------------------------------------------------------------
    //--------------------------------------------------------------

    ~ofxMSATensorFlow();

    // initialize session, return true if successfull
    bool setup(const tensorflow::SessionOptions & session_options = tensorflow::SessionOptions());


    // load a binary graph file and add to sessoin, return true if successfull
    bool loadGraph(const string path, tensorflow::Env* env = tensorflow::Env::Default());

    // create session with graph
    // called automatically by loadGraph,
    // you can also call it manually if you want to create or modify your own graph
    bool createSessionWithGraph();

    // run the graph with given inputs and outputs, return true if successfull
    // (more: https://www.tensorflow.org/versions/master/api_docs/cc/ClassSession.html#virtual_Status_tensorflow_Session_Run )
    bool run(const std::vector<std::pair<string, tensorflow::Tensor> >& inputs,
             const std::vector<string>& output_tensor_names,
             const std::vector<string>& target_node_names,
             std::vector<tensorflow::Tensor>* outputs);

    // get status of last action (more: https://www.tensorflow.org/versions/master/api_docs/cc/ClassStatus.html )
    const tensorflow::Status& status() const { return _status; }

    // get session (more: https://www.tensorflow.org/versions/master/api_docs/cc/ClassSession.html )
    tensorflow::Session* session() { return _session; }
    const tensorflow::Session* session() const { return _session; }

    // get graph. you can manipulate this, and then call createSessionWithGraph
    tensorflow::GraphDef& graph() { return _graph_def; }
    const tensorflow::GraphDef& graph() const { return _graph_def; }

    // get is ready or not (i.e. graph and session have been created successfully)
    bool isReady() const { return _initialized; }

protected:
    tensorflow::Session* _session = NULL;
    tensorflow::GraphDef _graph_def;
    tensorflow::Status _status;
    bool _initialized = false;

    void closeSession();
    void logError();
};

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


// return dimensions of an image to hold the Tensor: [x: width, y: height, z: depth (number of channels)]
ofVec3f ofxMSATensorFlow::tensorToPixelDims(const tensorflow::Tensor &t, string chmap) {
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
// flatten tensor and copy into std::vector. vector will be allocated if nessecary
template<typename T>
void ofxMSATensorFlow::tensorToVector(const tensorflow::Tensor &src, std::vector<T> &dst, bool do_memcpy) {
    if(dst.size() != src.NumElements()) dst.resize(src.NumElements());
    tensorToArray(src, dst.data(), do_memcpy);
}


//--------------------------------------------------------------
// copy into pixels. dst won't be reshaped if it's already allocated. otherwise it'll be allocated according to tensorPixelDims(, chmap)
template<typename T>
void ofxMSATensorFlow::tensorToPixels(const tensorflow::Tensor &src, ofPixels_<T> &dst, bool do_memcpy, string chmap) {
    if(!dst.isAllocated()) {
        ofVec3f dims(tensorToPixelDims(src, chmap));
        dst.allocate((int)dims.x, (int)dims.y, (int)dims.z);
    }

    tensorToArray(src, dst.getData(), do_memcpy);
}


//--------------------------------------------------------------
// copy into image. dst won't be reshaped if it's already allocated. otherwise it'll be allocated according to tensorPixelDims(, chmap)
template<typename T>
void ofxMSATensorFlow::tensorToImage(const tensorflow::Tensor &src, ofImage_<T> &dst, bool do_memcpy, string chmap) {
    if(!dst.isAllocated()) {
        ofVec3f dims(tensorToPixelDims(src, chmap));
        dst.allocate((int)dims.x, (int)dims.y, dims.z == 1 ? OF_IMAGE_GRAYSCALE : (dims.z == 3 ? OF_IMAGE_COLOR : OF_IMAGE_COLOR_ALPHA));
    }
    tensorToPixels(src, dst.getPixels(), do_memcpy);
    dst.update();
}


//--------------------------------------------------------------
// flatten tensor and copy into array. array must already be allocated
template<typename T>
void ofxMSATensorFlow::tensorToArray(const tensorflow::Tensor &src, T *dst, bool do_memcpy) {
    auto src_data = src.flat<T>().data();
    int n = src.NumElements();
    if(do_memcpy) memcpy(dst, src_data, n * sizeof(T));
    else for(int i=0; i<n; i++) dst[i] = src_data[i];
}


//--------------------------------------------------------------
// copy std::vector into flattened tensor. tensor must already be allocated and will not be reshaped
template<typename T>
void ofxMSATensorFlow::vectorToTensor(const std::vector<T> &src, tensorflow::Tensor &dst, bool do_memcpy) {
    arrayToTensor(src.data(), dst, do_memcpy);
}


//--------------------------------------------------------------
// copy pixels into tensor. tensor must already be allocated and will not be reshaped
template<typename T>
void ofxMSATensorFlow::pixelsToTensor(const ofPixels_<T> &src, tensorflow::Tensor &dst, bool do_memcpy) {
    arrayToTensor(src.getData(), dst, do_memcpy);
}


//--------------------------------------------------------------
// copy image into tensor. tensor must already be allocated and will not be reshaped
template<typename T>
void ofxMSATensorFlow::imageToTensor(const ofImage_<T> &src, tensorflow::Tensor &dst, bool do_memcpy) {
    pixelsToTensor(src.getPixels(), dst, do_memcpy);
}


//--------------------------------------------------------------
// copy array into flattened tensor. tensor must already be allocated and will not be reshaped
template<typename T>
void ofxMSATensorFlow::arrayToTensor(const T *in, tensorflow::Tensor &dst, bool do_memcpy) {
    auto dst_data = dst.flat<T>().data();
    int n = dst.NumElements();
    if(do_memcpy) memcpy(dst_data, in, n * sizeof(T));
    else for(int i=0; i<n; i++) dst_data[i] = in[i];
}


//--------------------------------------------------------------
// convert grayscale float image into RGB float image where R -ve and B is +ve
// dst image is allocated if nessecary
template<typename T>
void ofxMSATensorFlow::grayToColor(const ofPixels_<T> &src, ofPixels_<T> &dst, float scaler) {
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
template<typename T>
void ofxMSATensorFlow::grayToColor(const ofImage_<T> &src, ofImage_<T> &dst, float scaler) {
    grayToColor(src.getPixels(), dst.getPixels(), scaler);
    dst.update();
}


//--------------------------------------------------------------
void ofxMSATensorFlow::getTopScores(tensorflow::Tensor scores_tensor, int topk_count, vector<int> &out_indices, vector<float> &out_scores) {
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
