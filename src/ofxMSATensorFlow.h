/*
 this is simple wrapper for a single session and graph, should suffice for most cases
 you can access the internal variables directly if you need more advanced setup
 I didn't wrap it too much as I think it's important to understand how TensorFlow works,
 in case you need to switch to raw tensorflow project etc.

 There's also a bunch of helper functions for various functions (eg vector <--> tensor <--> image conversions)

*/

#pragma once

#include "ofxMSATFIncludes.h"
#include "ofxMSATFUtils.h"
#include "ofxMSATFVizUtils.h"
#include "ofxMSATFImageClassifier.h"
#include "ofxMSATFLayerVisualizer.h"
#include "ofxMSAMathUtils.h"

namespace msa {
namespace tf {



}   // namespace tf
}   // namespace msa
