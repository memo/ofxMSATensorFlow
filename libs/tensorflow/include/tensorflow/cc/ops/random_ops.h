// This file is MACHINE GENERATED! Do not edit.

#ifndef TENSORFLOW_CC_OPS_RANDOM_OPS_H_
#define TENSORFLOW_CC_OPS_RANDOM_OPS_H_

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


// Randomly shuffles a tensor along its first dimension.
//
//   The tensor is shuffled along dimension 0, such that each `value[j]` is mapped
//   to one and only one `output[i]`. For example, a mapping that might occur for a
//   3x2 tensor is:
//
// ```prettyprint
// [[1, 2],       [[5, 6],
//  [3, 4],  ==>   [1, 2],
//  [5, 6]]        [3, 4]]
// ```
//
// Arguments:
// * value: The tensor to be shuffled.
// * opts:
//   .WithAttr("seed", int64): Defaults to 0.
//     If either `seed` or `seed2` are set to be non-zero, the random number
// generator is seeded by the given seed.  Otherwise, it is seeded by a
// random seed.
//   .WithAttr("seed2", int64): Defaults to 0.
//     A second seed to avoid seed collision.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// A tensor of same shape and type as `value`, shuffled along its first
// dimension.
Node* RandomShuffle(NodeOut value, const GraphDefBuilder::Options& opts);

// Outputs random values from a normal distribution.
//
// The generated values will have mean 0 and standard deviation 1.
//
// Arguments:
// * shape: The shape of the output tensor.
// * dtype: The type of the output.
// * opts:
//   .WithAttr("seed", int64): Defaults to 0.
//     If either `seed` or `seed2` are set to be non-zero, the random number
// generator is seeded by the given seed.  Otherwise, it is seeded by a
// random seed.
//   .WithAttr("seed2", int64): Defaults to 0.
//     A second seed to avoid seed collision.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// A tensor of the specified shape filled with random normal values.
Node* RandomStandardNormal(NodeOut shape, DataType dtype, const
                           GraphDefBuilder::Options& opts);

// Outputs random values from a uniform distribution.
//
// The generated values follow a uniform distribution in the range `[0, 1)`. The
// lower bound 0 is included in the range, while the upper bound 1 is excluded.
//
// Arguments:
// * shape: The shape of the output tensor.
// * dtype: The type of the output.
// * opts:
//   .WithAttr("seed", int64): Defaults to 0.
//     If either `seed` or `seed2` are set to be non-zero, the random number
// generator is seeded by the given seed.  Otherwise, it is seeded by a
// random seed.
//   .WithAttr("seed2", int64): Defaults to 0.
//     A second seed to avoid seed collision.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// A tensor of the specified shape filled with uniform random values.
Node* RandomUniform(NodeOut shape, DataType dtype, const
                    GraphDefBuilder::Options& opts);

// Outputs random integers from a uniform distribution.
//
// The generated values are uniform integers in the range `[minval, maxval)`.
// The lower bound `minval` is included in the range, while the upper bound
// `maxval` is excluded.
//
// The random integers are slightly biased unless `maxval - minval` is an exact
// power of two.  The bias is small for values of `maxval - minval` significantly
// smaller than the range of the output (either `2^32` or `2^64`).
//
// Arguments:
// * shape: The shape of the output tensor.
// * minval: 0-D.  Inclusive lower bound on the generated integers.
// * maxval: 0-D.  Exclusive upper bound on the generated integers.
// * opts:
//   .WithAttr("seed", int64): Defaults to 0.
//     If either `seed` or `seed2` are set to be non-zero, the random number
// generator is seeded by the given seed.  Otherwise, it is seeded by a
// random seed.
//   .WithAttr("seed2", int64): Defaults to 0.
//     A second seed to avoid seed collision.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// A tensor of the specified shape filled with uniform random integers.
Node* RandomUniformInt(NodeOut shape, NodeOut minval, NodeOut maxval, const
                       GraphDefBuilder::Options& opts);

// Outputs random values from a truncated normal distribution.
//
// The generated values follow a normal distribution with mean 0 and standard
// deviation 1, except that values whose magnitude is more than 2 standard
// deviations from the mean are dropped and re-picked.
//
// Arguments:
// * shape: The shape of the output tensor.
// * dtype: The type of the output.
// * opts:
//   .WithAttr("seed", int64): Defaults to 0.
//     If either `seed` or `seed2` are set to be non-zero, the random number
// generator is seeded by the given seed.  Otherwise, it is seeded by a
// random seed.
//   .WithAttr("seed2", int64): Defaults to 0.
//     A second seed to avoid seed collision.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// A tensor of the specified shape filled with random truncated normal
// values.
Node* TruncatedNormal(NodeOut shape, DataType dtype, const
                      GraphDefBuilder::Options& opts);

}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_OPS_RANDOM_OPS_H_
