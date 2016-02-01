// This file is MACHINE GENERATED! Do not edit.

#ifndef TENSORFLOW_CC_OPS_LOGGING_OPS_H_
#define TENSORFLOW_CC_OPS_LOGGING_OPS_H_

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


// Asserts that the given condition is true.
//
// If `condition` evaluates to false, print the list of tensors in `data`.
// `summarize` determines how many entries of the tensors to print.
//
// Arguments:
// * condition: The condition to evaluate.
// * data: The tensors to print out when condition is false.
// * opts:
//   .WithAttr("summarize", int64): Defaults to 3.
//     Print this many entries of each tensor.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* Assert(NodeOut condition, gtl::ArraySlice<NodeOut> data, const
             GraphDefBuilder::Options& opts);

// Prints a list of tensors.
//
// Passes `input` through to `output` and prints `data` when evaluating.
//
// Arguments:
// * input: The tensor passed to `output`
// * data: A list of tensors to print out when op is evaluated.
// * opts:
//   .WithAttr("message", StringPiece): Defaults to "".
//     A string, prefix of the error message.
//   .WithAttr("first_n", int64): Defaults to -1.
//     Only log `first_n` number of times. -1 disables logging.
//   .WithAttr("summarize", int64): Defaults to 3.
//     Only print this many entries of each tensor.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The unmodified `input` tensor
Node* Print(NodeOut input, gtl::ArraySlice<NodeOut> data, const
            GraphDefBuilder::Options& opts);

}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_OPS_LOGGING_OPS_H_
