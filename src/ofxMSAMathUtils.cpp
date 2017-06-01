#include "ofxMSAMathUtils.h"


namespace msa {
namespace tf {

// calculate required scale and offset to map a number from range [input_min, input_max] to [output_min, output_max]
// pass in float variables which will be filled with the right values
// if scaling a massive array, this might be faster than using ofMap on every element
void calc_scale_offset(const ofVec2f& in_range, const ofVec2f& out_range, float& scale, float &offset) {
    calc_scale_offset(in_range.x, in_range.y, out_range.x, out_range.y, scale, offset);
}

void calc_scale_offset(float input_min, float input_max, float output_min, float output_max, float& scale, float &offset) {
    if(fabs(input_max - input_min) > std::numeric_limits<float>::epsilon()) {
        scale = 1.0f/(input_max - input_min) * (output_max - output_min);
        offset = (output_min - input_min * scale);
    } else {
        scale= 1.0f;
        offset = 0.0f;
    }
}


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
                           ) {

    ofVec2f ret;

    // sanity check all vectors have same size
    int k = o_pi.size();
    if(k == 0 || o_mu1.size() != k || o_mu2.size() != k || o_sigma1.size() != k || o_sigma2.size() != k || o_corr.size() != k) {
        ofLogWarning("ofxMSATensorFlow") << " sample_from_bi_gmm vector size mismatch ";
        return ret;
    }

    // two independent zero mean, unit variance gaussian variables
    std::normal_distribution<double> gaussian(0.0, 1.0);
    double z1 = gaussian(rng);
    double z2 = gaussian(rng);

    // pick mixture component to sample from
    int i = msa::tf::sample_from_prob(rng, o_pi);

    // transform with mu1, mu2, sigma1, sigma2, rho from i'th gaussian
    double mu1 = o_mu1[i];
    double mu2 = o_mu2[i];
    double sigma1 = o_sigma1[i];
    double sigma2 = o_sigma2[i];
    double rho = o_corr[i];

    ret.x = mu1 + sigma1 * z1;
    ret.y = mu2 + sigma2 * (z1*rho + z2*sqrt(1-rho*rho));
    return ret;
}

}
}
