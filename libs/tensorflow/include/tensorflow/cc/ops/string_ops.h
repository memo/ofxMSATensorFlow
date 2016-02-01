// This file is MACHINE GENERATED! Do not edit.

#ifndef TENSORFLOW_CC_OPS_STRING_OPS_H_
#define TENSORFLOW_CC_OPS_STRING_OPS_H_

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {
namespace ops {

// These add a node to the graph from opts.
//
// Note for "NodeOut" inputs, you will typically either pass
// * a {Node*, int index} (to pass the index-th output of that node), or
// * a Node* (to pass the first output of that node).


// Converts each string in the input Tensor to its hash mod by a number of buckets.
//
// The hash function is deterministic on the content of the string within the
// process.
//
// Note that the hash function may change from time to time.
//
// Arguments:
// * num_buckets: The number of buckets.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// A Tensor of the same shape as the input `string_tensor`.
Node* StringToHashBucket(NodeOut string_tensor, int64 num_buckets, const
                         GraphDefBuilder::Options& opts);

}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_OPS_STRING_OPS_H_
