#include "ofxMSAMathUtils.h"


namespace msa {
namespace tf {

// calculate required scale and offset to map a number from range [input_min, input_max] to [output_min, output_max]
// pass in float variables which will be filled with the right values
// if scaling a massive array, this might be faster than using ofMap on every element
// in the case of 'ranges', [input_min, input_max] = [ranges.x, ranges.y], [output_min, output_max] = [ranges.z, ranges.w]
void calc_scale_offset(const ofVec4f& ranges, float& scale, float &offset) {
    calc_scale_offset(ranges.x, ranges.y, ranges.z, ranges.w, scale, offset);
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
        ofLogWarning() << " sample_from_bi_gmm vector size mismatch ";
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
                      float scale    // arbitrary scale
        ) {
    // Correlation Matrix Sigma [[a b], [c d]]
    double a = sigma1*sigma1;
    double b = rho*sigma1*sigma2;
    double c = b;
    double d = sigma2*sigma2;

    double T = a+d; // trace
    double D = (a*d)-(b*c); // determinant

    // eigenvalues
    double l1 = T/2. + T*sqrt( 1./(4.-D) );
    double l2 = T/2. - T*sqrt( 1./(4.-D) );

    ofVec2f v1 ( ofVec2f(b, l1-a).normalized() );
    ofVec2f v2 ( ofVec2f(b, l2-a).normalized() );

    // create 4x4 transformation matrix
    // eigenvectors in upper left corner for rotation around z
    // scale diagonal by eigenvalues
    ofMatrix4x4 m44;
    m44.ofMatrix4x4::makeIdentityMatrix();
    m44.getPtr()[0] = v1.x * sqrtf(fabsf(l1)) * scale;
    m44.getPtr()[4] = v1.y;
    m44.getPtr()[1] = v2.x;
    m44.getPtr()[5] = v2.y * sqrtf(fabsf(l2)) * scale;
    m44.setTranslation(mu1, mu2, 0);

    ofPushMatrix();
    ofMultMatrix(m44);
    ofDrawCircle(0, 0, 1);
    ofPopMatrix();

    // trying raw opengl commands instead of ofXXX to make sure column and row order stuff is done as I want :S
    //    glPushMatrix();
    //    glMultMatrixf(m44.getPtr());
    //    ofDrawCircle(0, 0, 1);
    //    glPopMatrix();
}



//--------------------------------------------------------------
// visualise bivariate gaussian mixture model
void draw_bi_gmm(const vector<float>& o_pi,      // vector of mixture weights
                 const vector<float>& o_mu1,     // means 1
                 const vector<float>& o_mu2,     // means 2
                 const vector<float>& o_sigma1,  // sigmas 1
                 const vector<float>& o_sigma2,  // sigmas 2
                 const vector<float>& o_corr,    // correlations
                 const ofVec2f& offset,
                 float draw_scale,
                 float gaussian_scale,
                 ofColor color_min,
                 ofColor color_max
        ) {

    int k = o_pi.size();
    if(k == 0 || o_mu1.size() != k || o_mu2.size() != k || o_sigma1.size() != k || o_sigma2.size() != k || o_corr.size() != k) {
        ofLogWarning() << " draw_bi_gmm vector size mismatch ";
        return;
    }

    ofPushMatrix();
    ofTranslate(offset);
    ofScale(draw_scale, draw_scale);
    for(int i=0; i<k; i++) {
        ofColor c(color_min);
        c.lerp(color_max, o_pi[i]);
        ofSetColor(c);
        draw_bi_gaussian(o_mu1[i], o_mu2[i], o_sigma1[i], o_sigma2[i], o_corr[i], gaussian_scale);
    }
    ofPopMatrix();
}

}
}
