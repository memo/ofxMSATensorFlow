/*
 * General helper functions
 */

#pragma once

#include "ofxMSATFIncludes.h"

namespace msa {
namespace tf {

// draw probability histogram in target rect with colors lo_color for low probabilities, hi_color for high probabilities
void draw_probs(const vector<float>& probs, const ofRectangle& rect, const ofColor& lo_color=ofColor::blue, const ofColor& hi_color=ofColor::red);


}
}
