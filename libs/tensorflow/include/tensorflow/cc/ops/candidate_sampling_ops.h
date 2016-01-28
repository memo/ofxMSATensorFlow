// This file is MACHINE GENERATED! Do not edit.

#ifndef TENSORFLOW_CC_OPS_CANDIDATE_SAMPLING_OPS_H_
#define TENSORFLOW_CC_OPS_CANDIDATE_SAMPLING_OPS_H_

#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/public/tensor.h"
#include "tensorflow/core/public/tensor_shape.h"

namespace tensorflow {
namespace ops {

// These add a node to the graph from opts.
//
// Note for "NodeOut" inputs, you will typically either pass
// * a {Node*, int index} (to pass the index-th output of that node), or
// * a Node* (to pass the first output of that node).


// Generates labels for candidate sampling with a learned unigram distribution.
//
// See explanations of candidate sampling and the data formats at
// go/candidate-sampling.
//
// For each batch, this op picks a single set of sampled candidate labels.
//
// The advantages of sampling candidates per-batch are simplicity and the
// possibility of efficient dense matrix multiplication. The disadvantage is that
// the sampled candidates must be chosen independently of the context and of the
// true labels.
//
// Arguments:
// * true_classes: A batch_size * num_true matrix, in which each row contains the
// IDs of the num_true target_classes in the corresponding original label.
// * num_true: Number of true labels per context.
// * num_sampled: Number of candidates to produce per batch.
// * unique: If unique is true, we sample with rejection, so that all sampled
// candidates in a batch are unique. This requires some approximation to
// estimate the post-rejection sampling probabilities.
// * opts:
//   .WithAttr("seed", int64): Defaults to 0.
//     If either seed or seed2 are set to be non-zero, the random number
// generator is seeded by the given seed.  Otherwise, it is seeded by a
// random seed.
//   .WithAttr("seed2", int64): Defaults to 0.
//     An second seed to avoid seed collision.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with outputs:
// * sampled_candidates: A vector of length num_sampled, in which each element is
// the ID of a sampled candidate.
// * true_expected_count: A batch_size * num_true matrix, representing
// the number of times each candidate is expected to occur in a batch
// of sampled candidates. If unique=true, then this is a probability.
// * sampled_expected_count: A vector of length num_sampled, for each sampled
// candidate representing the number of times the candidate is expected
// to occur in a batch of sampled candidates.  If unique=true, then this is a
// probability.
Node* AllCandidateSampler(NodeOut true_classes, int64 num_true, int64
                          num_sampled, bool unique, const
                          GraphDefBuilder::Options& opts);

// Computes the ids of the positions in sampled_candidates that match true_labels.
//
// When doing log-odds NCE, the result of this op should be passed through a
// SparseToDense op, then added to the logits of the sampled candidates. This has
// the effect of 'removing' the sampled labels that match the true labels by
// making the classifier sure that they are sampled labels.
//
// Arguments:
// * true_classes: The true_classes output of UnpackSparseLabels.
// * sampled_candidates: The sampled_candidates output of CandidateSampler.
// * num_true: Number of true labels per context.
// * opts:
//   .WithAttr("seed", int64): Defaults to 0.
//     If either seed or seed2 are set to be non-zero, the random number
// generator is seeded by the given seed.  Otherwise, it is seeded by a
// random seed.
//   .WithAttr("seed2", int64): Defaults to 0.
//     An second seed to avoid seed collision.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with outputs:
// * indices: A vector of indices corresponding to rows of true_candidates.
// * ids: A vector of IDs of positions in sampled_candidates that match a true_label
// for the row with the corresponding index in indices.
// * weights: A vector of the same length as indices and ids, in which each element
// is -FLOAT_MAX.
Node* ComputeAccidentalHits(NodeOut true_classes, NodeOut sampled_candidates,
                            int64 num_true, const GraphDefBuilder::Options&
                            opts);

// Generates labels for candidate sampling with a learned unigram distribution.
//
// A unigram sampler could use a fixed unigram distribution read from a
// file or passed in as an in-memory array instead of building up the distribution
// from data on the fly. There is also an option to skew the distribution by
// applying a distortion power to the weights.
//
// The vocabulary file should be in CSV-like format, with the last field
// being the weight associated with the word.
//
// For each batch, this op picks a single set of sampled candidate labels.
//
// The advantages of sampling candidates per-batch are simplicity and the
// possibility of efficient dense matrix multiplication. The disadvantage is that
// the sampled candidates must be chosen independently of the context and of the
// true labels.
//
// Arguments:
// * true_classes: A batch_size * num_true matrix, in which each row contains the
// IDs of the num_true target_classes in the corresponding original label.
// * num_true: Number of true labels per context.
// * num_sampled: Number of candidates to randomly sample per batch.
// * unique: If unique is true, we sample with rejection, so that all sampled
// candidates in a batch are unique. This requires some approximation to
// estimate the post-rejection sampling probabilities.
// * range_max: The sampler will sample integers from the interval [0, range_max).
// * opts:
//   .WithAttr("vocab_file", StringPiece): Defaults to "".
//     Each valid line in this file (which should have a CSV-like format)
// corresponds to a valid word ID. IDs are in sequential order, starting from
// num_reserved_ids. The last entry in each line is expected to be a value
// corresponding to the count or relative probability. Exactly one of vocab_file
// and unigrams needs to be passed to this op.
//   .WithAttr("distortion", float): Defaults to 1.
//     The distortion is used to skew the unigram probability distribution.
// Each weight is first raised to the distortion's power before adding to the
// internal unigram distribution. As a result, distortion = 1.0 gives regular
// unigram sampling (as defined by the vocab file), and distortion = 0.0 gives
// a uniform distribution.
//   .WithAttr("num_reserved_ids", int64): Defaults to 0.
//     Optionally some reserved IDs can be added in the range [0,
// ..., num_reserved_ids) by the users. One use case is that a special unknown
// word token is used as ID 0. These IDs will have a sampling probability of 0.
//   .WithAttr("num_shards", int64): Defaults to 1.
//     A sampler can be used to sample from a subset of the original range
// in order to speed up the whole computation through parallelism. This parameter
// (together with 'shard') indicates the number of partitions that are being
// used in the overall computation.
//   .WithAttr("shard", int64): Defaults to 0.
//     A sampler can be used to sample from a subset of the original range
// in order to speed up the whole computation through parallelism. This parameter
// (together with 'num_shards') indicates the particular partition number of a
// sampler op, when partitioning is being used.
//   .WithAttr("unigrams", gtl::ArraySlice<float>): Defaults to [].
//     A list of unigram counts or probabilities, one per ID in sequential
// order. Exactly one of vocab_file and unigrams should be passed to this op.
//   .WithAttr("seed", int64): Defaults to 0.
//     If either seed or seed2 are set to be non-zero, the random number
// generator is seeded by the given seed.  Otherwise, it is seeded by a
// random seed.
//   .WithAttr("seed2", int64): Defaults to 0.
//     An second seed to avoid seed collision.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with outputs:
// * sampled_candidates: A vector of length num_sampled, in which each element is
// the ID of a sampled candidate.
// * true_expected_count: A batch_size * num_true matrix, representing
// the number of times each candidate is expected to occur in a batch
// of sampled candidates. If unique=true, then this is a probability.
// * sampled_expected_count: A vector of length num_sampled, for each sampled
// candidate representing the number of times the candidate is expected
// to occur in a batch of sampled candidates.  If unique=true, then this is a
// probability.
Node* FixedUnigramCandidateSampler(NodeOut true_classes, int64 num_true, int64
                                   num_sampled, bool unique, int64 range_max,
                                   const GraphDefBuilder::Options& opts);

// Generates labels for candidate sampling with a learned unigram distribution.
//
// See explanations of candidate sampling and the data formats at
// go/candidate-sampling.
//
// For each batch, this op picks a single set of sampled candidate labels.
//
// The advantages of sampling candidates per-batch are simplicity and the
// possibility of efficient dense matrix multiplication. The disadvantage is that
// the sampled candidates must be chosen independently of the context and of the
// true labels.
//
// Arguments:
// * true_classes: A batch_size * num_true matrix, in which each row contains the
// IDs of the num_true target_classes in the corresponding original label.
// * num_true: Number of true labels per context.
// * num_sampled: Number of candidates to randomly sample per batch.
// * unique: If unique is true, we sample with rejection, so that all sampled
// candidates in a batch are unique. This requires some approximation to
// estimate the post-rejection sampling probabilities.
// * range_max: The sampler will sample integers from the interval [0, range_max).
// * opts:
//   .WithAttr("seed", int64): Defaults to 0.
//     If either seed or seed2 are set to be non-zero, the random number
// generator is seeded by the given seed.  Otherwise, it is seeded by a
// random seed.
//   .WithAttr("seed2", int64): Defaults to 0.
//     An second seed to avoid seed collision.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with outputs:
// * sampled_candidates: A vector of length num_sampled, in which each element is
// the ID of a sampled candidate.
// * true_expected_count: A batch_size * num_true matrix, representing
// the number of times each candidate is expected to occur in a batch
// of sampled candidates. If unique=true, then this is a probability.
// * sampled_expected_count: A vector of length num_sampled, for each sampled
// candidate representing the number of times the candidate is expected
// to occur in a batch of sampled candidates.  If unique=true, then this is a
// probability.
Node* LearnedUnigramCandidateSampler(NodeOut true_classes, int64 num_true,
                                     int64 num_sampled, bool unique, int64
                                     range_max, const GraphDefBuilder::Options&
                                     opts);

// Generates labels for candidate sampling with a log-uniform distribution.
//
// See explanations of candidate sampling and the data formats at
// go/candidate-sampling.
//
// For each batch, this op picks a single set of sampled candidate labels.
//
// The advantages of sampling candidates per-batch are simplicity and the
// possibility of efficient dense matrix multiplication. The disadvantage is that
// the sampled candidates must be chosen independently of the context and of the
// true labels.
//
// Arguments:
// * true_classes: A batch_size * num_true matrix, in which each row contains the
// IDs of the num_true target_classes in the corresponding original label.
// * num_true: Number of true labels per context.
// * num_sampled: Number of candidates to randomly sample per batch.
// * unique: If unique is true, we sample with rejection, so that all sampled
// candidates in a batch are unique. This requires some approximation to
// estimate the post-rejection sampling probabilities.
// * range_max: The sampler will sample integers from the interval [0, range_max).
// * opts:
//   .WithAttr("seed", int64): Defaults to 0.
//     If either seed or seed2 are set to be non-zero, the random number
// generator is seeded by the given seed.  Otherwise, it is seeded by a
// random seed.
//   .WithAttr("seed2", int64): Defaults to 0.
//     An second seed to avoid seed collision.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with outputs:
// * sampled_candidates: A vector of length num_sampled, in which each element is
// the ID of a sampled candidate.
// * true_expected_count: A batch_size * num_true matrix, representing
// the number of times each candidate is expected to occur in a batch
// of sampled candidates. If unique=true, then this is a probability.
// * sampled_expected_count: A vector of length num_sampled, for each sampled
// candidate representing the number of times the candidate is expected
// to occur in a batch of sampled candidates.  If unique=true, then this is a
// probability.
Node* LogUniformCandidateSampler(NodeOut true_classes, int64 num_true, int64
                                 num_sampled, bool unique, int64 range_max,
                                 const GraphDefBuilder::Options& opts);

// Generates labels for candidate sampling with a learned unigram distribution.
//
// See explanations of candidate sampling and the data formats at
// go/candidate-sampling.
//
// For each batch, this op picks a single set of sampled candidate labels.
//
// The advantages of sampling candidates per-batch are simplicity and the
// possibility of efficient dense matrix multiplication. The disadvantage is that
// the sampled candidates must be chosen independently of the context and of the
// true labels.
//
// Arguments:
// * true_classes: A batch_size * num_true matrix, in which each row contains the
// IDs of the num_true target_classes in the corresponding original label.
// * num_true: Number of true labels per context.
// * num_sampled: Number of candidates to randomly sample per batch.
// * unique: If unique is true, we sample with rejection, so that all sampled
// candidates in a batch are unique. This requires some approximation to
// estimate the post-rejection sampling probabilities.
// * range_max: The sampler will sample integers from the interval [0, range_max).
// * opts:
//   .WithAttr("seed", int64): Defaults to 0.
//     If either seed or seed2 are set to be non-zero, the random number
// generator is seeded by the given seed.  Otherwise, it is seeded by a
// random seed.
//   .WithAttr("seed2", int64): Defaults to 0.
//     An second seed to avoid seed collision.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with outputs:
// * sampled_candidates: A vector of length num_sampled, in which each element is
// the ID of a sampled candidate.
// * true_expected_count: A batch_size * num_true matrix, representing
// the number of times each candidate is expected to occur in a batch
// of sampled candidates. If unique=true, then this is a probability.
// * sampled_expected_count: A vector of length num_sampled, for each sampled
// candidate representing the number of times the candidate is expected
// to occur in a batch of sampled candidates.  If unique=true, then this is a
// probability.
Node* ThreadUnsafeUnigramCandidateSampler(NodeOut true_classes, int64 num_true,
                                          int64 num_sampled, bool unique, int64
                                          range_max, const
                                          GraphDefBuilder::Options& opts);

// Generates labels for candidate sampling with a uniform distribution.
//
// See explanations of candidate sampling and the data formats at
// go/candidate-sampling.
//
// For each batch, this op picks a single set of sampled candidate labels.
//
// The advantages of sampling candidates per-batch are simplicity and the
// possibility of efficient dense matrix multiplication. The disadvantage is that
// the sampled candidates must be chosen independently of the context and of the
// true labels.
//
// Arguments:
// * true_classes: A batch_size * num_true matrix, in which each row contains the
// IDs of the num_true target_classes in the corresponding original label.
// * num_true: Number of true labels per context.
// * num_sampled: Number of candidates to randomly sample per batch.
// * unique: If unique is true, we sample with rejection, so that all sampled
// candidates in a batch are unique. This requires some approximation to
// estimate the post-rejection sampling probabilities.
// * range_max: The sampler will sample integers from the interval [0, range_max).
// * opts:
//   .WithAttr("seed", int64): Defaults to 0.
//     If either seed or seed2 are set to be non-zero, the random number
// generator is seeded by the given seed.  Otherwise, it is seeded by a
// random seed.
//   .WithAttr("seed2", int64): Defaults to 0.
//     An second seed to avoid seed collision.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with outputs:
// * sampled_candidates: A vector of length num_sampled, in which each element is
// the ID of a sampled candidate.
// * true_expected_count: A batch_size * num_true matrix, representing
// the number of times each candidate is expected to occur in a batch
// of sampled candidates. If unique=true, then this is a probability.
// * sampled_expected_count: A vector of length num_sampled, for each sampled
// candidate representing the number of times the candidate is expected
// to occur in a batch of sampled candidates.  If unique=true, then this is a
// probability.
Node* UniformCandidateSampler(NodeOut true_classes, int64 num_true, int64
                              num_sampled, bool unique, int64 range_max, const
                              GraphDefBuilder::Options& opts);

}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_OPS_CANDIDATE_SAMPLING_OPS_H_
