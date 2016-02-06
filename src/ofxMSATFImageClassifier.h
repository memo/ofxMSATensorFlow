#pragma once

#include "ofxMSATFIncludes.h"

namespace msa {
namespace tf {

class ofxMSATensorFlow;

class ImageClassifier {
public:
    struct Settings {
        string model_path;              // path to model file
        string labels_path;             // if exists, path to labels file (can be left empty)
        string input_layer_name;        // layer name of inputs
        string output_layer_name;       // layer name of outputs
        string dropout_layer_name;      // if exists, layer name of dropout keep probability (need to set to 1.0 during feedforward)
        string varconst_layer_suffix;   // if exists, suffix for CONSTANT layers with values for variables (see 'hack_variables' below)
        float norm_mean = 0;            // mean of all samples, used for normalization (0...1)
        float norm_stddev = 0;          // stddev of all samples, used for normalization (0...1). leave as 0 if you don't want normalization to happen
        std::vector<tensorflow::int64> image_dims;      // dimensions of input image. must have 3 values: { w, h, c (number of channels) }
        std::vector<tensorflow::int64> itensor_dims;    // dimensions of tensor to pass to tensorflow (could be same as image, could be shaped different, depends on network architecture)
    } settings;


    ImageClassifier()                                           {}
    ImageClassifier(const Settings& settings)                   { setup(settings); }


    bool setup(const Settings& settings);
    bool isReady() const;


    // Load pixels into the network, get the results
    bool classify(const ofPixels &pix);

    // if you disable gltextures of the images, performance can be a little better
    void setUseTexture(bool b);

    //--------------------------------------------------------------
    // GETTERS

    ofxMSATensorFlow& getMsatf()                                { return *msa_tf; }
    const ofxMSATensorFlow& getMsatf() const                    { return *msa_tf; }

    vector<tensorflow::Tensor>& getOutputTensors()              { return output_tensors; }
    const vector<tensorflow::Tensor>& getOutputTensors() const  { return output_tensors; }

    ofImage& getInputImage()                                    { return input_img; }
    const ofImage& getInputImage() const                        { return input_img; }

    ofFloatImage& getProcessedImage()                           { return processed_img; }
    const ofFloatImage& getProcessedImage() const               { return processed_img; }

    // returns all probabilities for all classes. Can be very large! (e.g. 1000, for imagenet)
    // use get_top_XXX if you just want the top few
    vector<float>& getClassProbs()                              { return class_probs; }
    const vector<float>& getClassProbs() const                  { return class_probs; }

    // returns all labels. i.e. as read from file)
    vector<string>& getLabels()                                 { return labels; }
    const vector<string>& getLabels() const                     { return labels; }

    // total number of classes
    int getNumClasses() const                                   { return class_probs.size(); }


    int getWidth() const                                        { return settings.image_dims[0]; }
    int getHeight() const                                       { return settings.image_dims[1]; }
    int getDepth() const                                        { return settings.image_dims[2]; }


protected:
    shared_ptr<ofxMSATensorFlow> msa_tf;            // interface to tensorflow
    tensorflow::Tensor  image_tensor;               // stores input image as tensor
    vector<tensorflow::Tensor> output_tensors;      // stores all output tensors

    ofImage input_img;              // original input image
    ofFloatImage processed_img;     // normalized float version of input image, using Image instead of Pix because Pix has shit resizing!

    // contains classification information from last classification attempt
    vector<string> labels;          // contains all labels
    vector<float> class_probs;      // contains all probabilities;

    int num_elements;               // total number of elements in image or tensor (w x h x c)


    //---------------------------------------------------------
    // MEGA UGLY HACK ALERT
    // graphs exported from python don't store values for trained variables (i.e. parameters)
    // so in python we need to add the values back to the graph as 'constants'
    // and bang them here to push values to parameters
    // more here: https://stackoverflow.com/questions/34343259/is-there-an-example-on-how-to-generate-protobuf-files-holding-trained-tensorflow/34343517
    // assuming the variables in the model have constants attached to them with a substr in their name
    bool hack_variables(string substr);
};

}   // namespace tf
}   // namespace msa
