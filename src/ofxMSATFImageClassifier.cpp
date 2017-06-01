#include "ofxMSATFImageClassifier.h"
#include "ofxMSATensorFlow.h"

namespace msa {
namespace tf {


//--------------------------------------------------------------
ImageClassifier::~ImageClassifier() {
    if(session) session->Close();
}


//--------------------------------------------------------------
bool ImageClassifier::isReady() const {
    return session != nullptr;
}


//--------------------------------------------------------------
void ImageClassifier::setUseTexture(bool b) {
    processed_img.setUseTexture(b);
    input_img.setUseTexture(b);
}


//--------------------------------------------------------------
bool ImageClassifier::setup(const ImageClassifier::Settings& settings, string device) {
    // close existing session
    if(session) session->Close();

    this->settings = settings;

    // make sure we have rank==3 for image dims (w, h, c)
    if(settings.image_dims.size() != 3) {
        ofLogError("ofxMSATensorFlow") << "ImageClassifier::ImageClassifier -image dimesions needs three: width x height x number of channels";
        return false;
    }

    // calculate total number of elements in input image
    num_elements = 1;
    for(auto i : settings.image_dims) num_elements *= i;

    // calculate total number of elements in input tensor
    int test_num_elements = 1;
    for(auto i : settings.itensor_dims) test_num_elements *= i;

    // sanitiy check, make sure they match, return if error
    if(test_num_elements != num_elements) {
        ofLogError("ofxMSATensorFlow") << "ImageClassifier::ImageClassifier - Image elements doesn't match tensor elements:" << num_elements << " != " << test_num_elements;
        num_elements = 0;
        return false;
    }

    // Load graph (i.e. trained model), return if error
    graph_def = msa::tf::load_graph_def(settings.model_path);
    if(!graph_def) return false;

    // Initialize tensorflow session with graph, return if error
    session = msa::tf::create_session_with_graph(graph_def, device);
    if(!session) return false;

    // load text file containing labels (i.e. associating classification index with human readable text)
    if(!settings.labels_path.empty()) if( !read_labels_file(ofToDataPath(settings.labels_path), labels) ) return false;

    // initialize input tensor dimensions
    // (not sure what the best way to do this was as there isn't an 'init' method, just a constructor)
    image_tensor = tensorflow::Tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(settings.itensor_dims));

    // constant -> variable hack
    if(!settings.varconst_layer_suffix.empty()) return hack_variables(settings.varconst_layer_suffix);

    return true;
}


//---------------------------------------------------------
void ImageClassifier::draw_probs(const ofRectangle& rect, const ofColor& lo_color, const ofColor& hi_color) {
    msa::tf::draw_probs(class_probs, rect, lo_color, hi_color);
}


//---------------------------------------------------------
// MEGA UGLY HACK ALERT
// graphs exported from python don't store values for trained variables (i.e. parameters)
// so in python we need to add the values back to the graph as 'constants'
// and bang them here to push values to parameters
// more here: https://stackoverflow.com/questions/34343259/is-there-an-example-on-how-to-generate-protobuf-files-holding-trained-tensorflow/34343517
// assuming the variables in the model have constants attached to them with a substr in their name
// TODO: get rid of this and update examples to use freeze_graph.py
bool ImageClassifier::hack_variables(string substr) {
    // will store names of constant units
    std::vector<string> names;

    int node_count = graph_def->node_size();
    ofLogNotice("ofxMSATensorFlow") << "Classifier::hack_variables - " << node_count << " nodes in graph";

    // iterate all nodes
    for(int i=0; i<node_count; i++) {
        auto n = graph_def->node(i);
        ofLogNotice("ofxMSATensorFlow") << i << ":" << n.name();

        // if name contains var_hack, add to vector
        if(n.name().find(substr) != std::string::npos) {
            names.push_back(n.name());
            ofLogNotice("ofxMSATensorFlow") << "......bang";
        }
    }

    // run the network inputting the names of the constant variables we want to run
    if(!session->Run({}, names, {}, &output_tensors).ok()) {
        ofLogError("ofxMSATensorFlow") << "Error running network for weights and biases variable hack";
        return false;
    }
    return true;
}

//---------------------------------------------------------
// Load pixels into the network, get the results
bool ImageClassifier::classify(const ofPixels &pix)  {
    input_img.setFromPixels(pix);

    if(!isReady()) return false;

    int iw = settings.image_dims[0];
    int ih = settings.image_dims[1];
    int iz = settings.image_dims[2];

    // convert to float
    ofFloatPixels fpix = pix;

    // set number of channels
    fpix.setNumChannels(iz);

    processed_img.setFromPixels(fpix);

    // resize (ofPixels::resize is crap, so using ofImage
    processed_img.resize(iw, ih);

    // pixelwise normalize image by subtracting the mean and dividing by stddev (across entire dataset)
    // I could do this without iterating over the pixels, by setting up a TensorFlow Graph, but I can't be bothered, this is less code
    if(settings.norm_stddev > 0) {
        float* pix_data = processed_img.getPixels().getData();
        if(!pix_data) {
            ofLogError("ofxMSATensorFlow") << "Could not classify. pixel data is NULL";
            return false;
        }
        for(int i=0; i<num_elements; i++) pix_data[i] = (pix_data[i] - settings.norm_mean) / settings.norm_stddev;
    }

    // copy data from image into tensorflow's Tensor class
    msa::tf::image_to_tensor(processed_img, image_tensor);

    // Collect inputs into a vector
    // IMPORTANT: the string must match the name of the variable/node in the graph
    vector<pair<string, tensorflow::Tensor>> inputs = { { settings.input_layer_name, image_tensor } };


    // if exists, set dropout probability to 1, not sure if this is the best way or if it can be disabled before saving the model
    if(!settings.dropout_layer_name.empty()) {
        tensorflow::Tensor keep_prob(tensorflow::DT_FLOAT, tensorflow::TensorShape());
        keep_prob.scalar<float>()() = 1.0f;
        inputs.push_back({settings.dropout_layer_name, keep_prob});
    }

    // feed the data into the network, and request output
    // output_tensors don't need to be initialized or allocated. they will be filled once the network runs
    if(!session->Run(inputs, { settings.output_layer_name }, {}, &output_tensors).ok()) {
        ofLogError("ofxMSATensorFlow") << "Error during running. Check console for details.";
        return false;
    }

    // copy from tensor to a vector using Mega Super Awesome convenience methods, because it's easy to deal with :)
    // output_tensors[0] contains the main outputs tensor
    msa::tf::tensor_to_vector(output_tensors[0], class_probs);

    return true;
}


}   // namespace tf
}   // namespace msa
