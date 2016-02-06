#include "ofxMSATFLayerVisualizer.h"
#include "ofxMSATensorFlow.h"

namespace msa {
namespace tf {


void LayerVisualizer::setup(ofxMSATensorFlow& msa_tf, string viz_layer_suffix) {
    bool do_visualize_nodes = true;
    if(do_visualize_nodes) {
        // first find the layer names that have viz_layer_suffix in them
        std::vector<string> names;
        int node_count = msa_tf.graph().node_size();

        // iterate all nodes
        for(int i=0; i<node_count; i++) {
            auto n = msa_tf.graph().node(i);
            if(n.name().find(viz_layer_suffix) != std::string::npos) {
                names.push_back(n.name());
            }
        }

        // lets get the weights from the network to visualize them in images
        // run the network and ask for nodes with the names selected above
        vector<tensorflow::Tensor> output_tensors;
        if( !msa_tf.run({}, names, {}, &output_tensors)) ofLogError() << "Error running network to get viz layers";

        int nlayers = output_tensors.size();    // number of layers in network

        weight_imgs.resize(nlayers);

        for(int l=0; l<nlayers; l++) {
            // weights matrix is a bit awkward
            // each column contains flattened weights for each pixel of the input image for a particular digit
            // i.e. 10 columns (one per digit) and 784 rows (one for each pixel of the input image)
            // we need to transpose the weights matrix to easily get sections of it out, this is easy as an image
            ofFloatPixels weights_pix_full;  // rows: weights for each digit (10), col: weights for each pixel (784)
            msa::tf::tensorToPixels(output_tensors[l], weights_pix_full, false, "10");
            weights_pix_full.rotate90(1);   // now rows: weights for each pixel, cols: weights for each digit
            weights_pix_full.mirror(false, true);

            int nunits = weights_pix_full.getHeight();  // number of units in layer
            int npixels = weights_pix_full.getWidth();    // number of pixels per unit
            int img_size = sqrt(npixels);                 // size of image (i.e. sqrt of above)
            weight_imgs[l].resize(nunits);

            ofFloatImage timg; // temp single channel image
            for(int i=0; i<nunits; i++) {
                weight_imgs[l][i] = make_shared<ofFloatImage>();

                // get data from full weights matrix into a single channel image
                int row_offset = i * npixels;
                timg.setFromPixels(weights_pix_full.getData() + row_offset, img_size, img_size, OF_IMAGE_COLOR);

                // convert single channel image into rgb (R showing -ve weights, B showing +ve weights)
                float scaler = nlayers * nunits * 0.1; // arbitrary scaler to work with both shallow and deep model
                msa::tf::grayToColor(timg, *weight_imgs[l][i], scaler);
            }
        }
    }

}


float LayerVisualizer::draw(float x, float y, float w, float padding) const {
    int nlayers = weight_imgs.size();
    float y_orig = y;
    for(int l=nlayers-1; l>=0; l--) {
        int nnodes = weight_imgs[l].size();
        //nnodes = min(nnodes, 32);   // put a cap on how many to draw?
        float s = w / nnodes;
        float ss = s * (1 - padding);
        for(int i=0; i<nnodes; i++) weight_imgs[l][i]->draw(x + i*s, y, ss, ss);
        y += s;
    }
    return y - y_orig;  // return height
}


}   // namespace tf
}   // namespace msa
