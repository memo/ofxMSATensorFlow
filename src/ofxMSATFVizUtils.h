/*
 * General helper functions
 */

#pragma once

#include "ofxMSATFIncludes.h"

namespace msa {
namespace tf {

// draw probability histogram in target rect with colors lo_color for low probabilities, hi_color for high probabilities
void draw_probs(const vector<float>& probs, const ofRectangle& rect, const ofColor& lo_color=ofColor::blue, const ofColor& hi_color=ofColor::red);

//--------------------------------------------------------------
// visualise bivariate gaussian distribution as an ellipse
// rotate unit circle by matrix of normalised eigenvectors and scale by sqrt eigenvalues (tip from @colormotor)
// eigen decomposition from on http://www.math.harvard.edu/archive/21b_fall_04/exhibits/2dmatrices/index.html
// also see nice visualisation at http://demonstrations.wolfram.com/TheBivariateNormalDistribution/
void draw_bi_gaussian(float mu1,     // mean 1
                      float mu2,     // mean 2
                      float sigma1,  // sigma 1
                      float sigma2,  // sigma 2
                      float rho,     // correlation
                      float scale=1.0 // arbitrary scale
        );

//--------------------------------------------------------------
// visualise bivariate gaussian mixture model
void draw_bi_gmm(const vector<float>& o_pi,      // vector of mixture weights
                 const vector<float>& o_mu1,     // means 1
                 const vector<float>& o_mu2,     // means 2
                 const vector<float>& o_sigma1,  // sigmas 1
                 const vector<float>& o_sigma2,  // sigmas 2
                 const vector<float>& o_corr,    // correlations
                 const ofVec2f& offset=ofVec2f::zero(),
                 float draw_scale=1.0,
                 float gaussian_scale=1.0,
                 ofColor color_min=ofColor(0, 200, 0, 20),
                 ofColor color_max=ofColor(0, 200, 0, 100)
        );

}
}
