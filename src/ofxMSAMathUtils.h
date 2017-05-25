#pragma once

#include "ofMain.h"
#include <random>

namespace msa {
namespace tf {


// calculate required scale and offset to map a number from range [input_min, input_max] to [output_min, output_max]
// pass in float variables which will be filled with the right values
// if scaling a massive array, this might be faster than using ofMap on every element
void calc_scale_offset(const ofVec2f& in_range, const ofVec2f& out_range, float& scale, float &offset);
void calc_scale_offset(float input_min, float input_max, float output_min, float output_max, float& scale, float &offset);

// adjust probability distribution with temperature (bias)
template<typename T> vector<T> adjust_probs_with_temp(const vector<T>& p_in, float t);

// sample from probability distribution
template<typename T> int sample_from_prob(std::default_random_engine& rgen, const vector<T>& p);

// zero vector
template<typename T> void zero_probs(vector<T>& p);


//--------------------------------------------------------------
// sample a point from a bivariate gaussian mixture model
// maths based on http://www.statisticalengineering.com/bivariate_normal.htm
ofVec2f sample_from_bi_gmm(std::default_random_engine& rng,// random number generator
                           const vector<float>& o_pi,      // vector of mixture weights
                           const vector<float>& o_mu1,     // means 1
                           const vector<float>& o_mu2,     // means 2
                           const vector<float>& o_sigma1,  // sigmas 1
                           const vector<float>& o_sigma2,  // sigmas 2
                           const vector<float>& o_corr     // correlations
                           );


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
template<typename T> vector<T> adjust_probs_with_temp(const vector<T>& p_in, float t) {
    if(t>0) {
        vector<T> p_out(p_in.size());
        T sum = 0;
        for(size_t i=0; i<p_in.size(); i++) {
            p_out[i] = exp( log((double)p_in[i]) / (double)t );
            sum += p_out[i];
        }

        if(sum > 0)
            for(size_t i=0; i<p_out.size(); i++) p_out[i] /= sum;

        return p_out;
    }

    return p_in;
}



//--------------------------------------------------------------
template<typename T> int sample_from_prob(std::default_random_engine& rng, const vector<T>& p) {
    std::discrete_distribution<int> rdist (p.begin(),p.end());
    int r = rdist(rng);
    return r;
}



//--------------------------------------------------------------
template<typename T> void zero_probs(vector<T>& p) {
    for(auto&& f : p) f = 0;
}


}
}
