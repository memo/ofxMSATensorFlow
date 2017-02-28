#pragma once

#include "ofMain.h"
#include <random>

namespace msa {
namespace tf {


// adjust probability distribution with temperature (bias)
template<typename T> vector<T> adjust_probs_with_temp(const vector<T>& p_in, float t);

// sample from probability distribution
template<typename T> int sample_from_prob(std::default_random_engine& rgen, const vector<T>& p);

// zero vector
template<typename T> void zero_probs(vector<T>& p);



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
