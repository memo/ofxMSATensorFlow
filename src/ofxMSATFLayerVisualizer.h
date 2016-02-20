/*
    very simple visualization of layers
    only really meaningful on single layer models,
    deeper networks need more complex visualization ( see http://arxiv.org/abs/1311.2901 )

    only tested on MNIST. needs testing on other models

 */

#pragma once

#include "ofxMSATFIncludes.h"
#include "ofxMSATFUtils.h"

namespace msa {
namespace tf {

class ofxMSATensorFlow;

class LayerVisualizer {
public:

    // pass instance of session/graph to visualize
    // visualizes all layers with viz_layer_suffix in the layer name
    void setup(tensorflow::Session& session, const tensorflow::GraphDef& graph_def, string viz_layer_suffix);

    void setup(Session_ptr session, const GraphDef_ptr graph_def, string viz_layer_suffix) { setup(*session, *graph_def, viz_layer_suffix); }

    // draw all layers at (x,y) with total width w, and padding
    // return total height of whatever is drawn
    float draw(float x, float y, float w, float padding = 0.1) const;

protected:
    // will use for visualizing weights.
    // inner storage is for each node of layer, outer storage is for layers
    vector< vector< std::shared_ptr<ofFloatImage> > > weight_imgs;
    //    vector< std::shared_ptr<ofFloatImage> > bias_imgs;

};


}   // namespace tf
}   // namespace msa
