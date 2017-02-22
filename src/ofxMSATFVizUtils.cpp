#include "ofxMSATFVizUtils.h"

namespace msa {
namespace tf {

void draw_probs(const vector<float>& probs, const ofRectangle& rect, const ofColor& lo_color, const ofColor& hi_color) {
    if(probs.size() == 0) return;

    ofPushStyle();
    ofFill();
    ofRectangle r(rect);
    r.width = rect.width / probs.size();
    for(int i=0; i<probs.size(); i++) {
        float p = probs[i];
        r.height = p * rect.height;
        r.y = rect.getBottom() - r.height;
        r.x += r.width;
        ofSetColor(lo_color + (hi_color - lo_color)* p);
        ofDrawRectangle(r);
    }
    ofPopStyle();
}

}
}
