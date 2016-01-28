// This file is MACHINE GENERATED! Do not edit.

#ifndef TENSORFLOW_CC_OPS_CONTROL_FLOW_OPS_H_
#define TENSORFLOW_CC_OPS_CONTROL_FLOW_OPS_H_

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


// Does nothing. Serves as a control trigger for scheduling. Only useful as a
//
// placeholder for control edges.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* ControlTrigger(const GraphDefBuilder::Options& opts);

// Creates or finds a child frame, and makes `data` available to the child frame.
//
// This op is used together with `Exit` to create loops in the graph.
// The unique `frame_name` is used by the `Executor` to identify frames. If
// `is_constant` is true, `output` is a constant in the child frame; otherwise
// it may be changed in the child frame. At most `parallel_iterations` iterations
// are run in parallel in the child frame.
//
// Arguments:
// * data: The tensor to be made available to the child frame.
// * frame_name: The name of the child frame.
// * opts:
//   .WithAttr("is_constant", bool): Defaults to false.
//     If true, the output is constant within the child frame.
//   .WithAttr("parallel_iterations", int64): Defaults to 10.
//     The number of iterations allowed to run in parallel.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The same tensor as `data`.
Node* Enter(NodeOut data, StringPiece frame_name, const
            GraphDefBuilder::Options& opts);

// Exits the current frame to its parent frame.
//
// Exit makes its input `data` available to the parent frame.
//
// Arguments:
// * data: The tensor to be made available to the parent frame.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The same tensor as `data`.
Node* Exit(NodeOut data, const GraphDefBuilder::Options& opts);

// Forwards the input to the output.
//
// This operator represents the loop termination condition used by the
// "pivot" switches of a loop.
//
// Arguments:
// * input: A boolean scalar, representing the branch predicate of the Switch op.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The same tensor as `input`.
Node* LoopCond(NodeOut input, const GraphDefBuilder::Options& opts);

// Forwards the value of an available tensor from `inputs` to `output`.
//
// `Merge` waits for at least one of the tensors in `inputs` to become available.
// It is usually combined with `Switch` to implement branching.
//
// `Merge` forwards the first tensor for become available to `output`, and sets
// `value_index` to its index in `inputs`.
//
// It is an error if more than one tensor in `inputs` is available.
//
// Arguments:
// * inputs: The input tensors, exactly one of which will become available.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with outputs:
// * output: Will be set to the available input tensor.
// * value_index: The index of the chosen input tensor in `inputs`.
Node* Merge(gtl::ArraySlice<NodeOut> inputs, const GraphDefBuilder::Options&
            opts);

// Makes its input available to the next iteration.
//
// Arguments:
// * data: The tensor to be made available to the next iteration.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The same tensor as `data`.
Node* NextIteration(NodeOut data, const GraphDefBuilder::Options& opts);

// Creates or finds a child frame, and makes `data` available to the child frame.
//
// The unique `frame_name` is used by the `Executor` to identify frames. If
// `is_constant` is true, `output` is a constant in the child frame; otherwise
// it may be changed in the child frame. At most `parallel_iterations` iterations
// are run in parallel in the child frame.
//
// Arguments:
// * data: The tensor to be made available to the child frame.
// * frame_name: The name of the child frame.
// * opts:
//   .WithAttr("is_constant", bool): Defaults to false.
//     If true, the output is constant within the child frame.
//   .WithAttr("parallel_iterations", int64): Defaults to 10.
//     The number of iterations allowed to run in parallel.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The same tensor as `data`.
Node* RefEnter(NodeOut data, StringPiece frame_name, const
               GraphDefBuilder::Options& opts);

// Exits the current frame to its parent frame.
//
// Exit makes its input `data` available to the parent frame.
//
// Arguments:
// * data: The tensor to be made available to the parent frame.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The same tensor as `data`.
Node* RefExit(NodeOut data, const GraphDefBuilder::Options& opts);

// Forwards the value of an available tensor from `inputs` to `output`.
//
// `Merge` waits for at least one of the tensors in `inputs` to become available.
// It is usually combined with `Switch` to implement branching.
//
// `Merge` forwards the first tensor for become available to `output`, and sets
// `value_index` to its index in `inputs`.
//
// It is an error if more than one tensor in `inputs` is available.
//
// Arguments:
// * inputs: The input tensors, exactly one of which will become available.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with outputs:
// * output: Will be set to the available input tensor.
// * value_index: The index of the chosen input tensor in `inputs`.
Node* RefMerge(gtl::ArraySlice<NodeOut> inputs, const GraphDefBuilder::Options&
               opts);

// Makes its input available to the next iteration.
//
// Arguments:
// * data: The tensor to be made available to the next iteration.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The same tensor as `data`.
Node* RefNextIteration(NodeOut data, const GraphDefBuilder::Options& opts);

// Forwards the `index`th element of `inputs` to `output`.
//
// Arguments:
// * index: A scalar that determines the input that gets selected.
// * inputs: A list of ref tensors, one of which will be forwarded to `output`.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The forwarded tensor.
Node* RefSelect(NodeOut index, gtl::ArraySlice<NodeOut> inputs, const
                GraphDefBuilder::Options& opts);

// Forwards the ref tensor `data` to the output port determined by `pred`.
//
// If `pred` is true, the `data` input is forwarded to `output_true`. Otherwise,
// the data goes to `output_false`.
//
// See also `Switch` and `Merge`.
//
// Arguments:
// * data: The ref tensor to be forwarded to the appropriate output.
// * pred: A scalar that specifies which output port will receive data.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with outputs:
// * output_false: If `pred` is false, data will be forwarded to this output.
// * output_true: If `pred` is true, data will be forwarded to this output.
Node* RefSwitch(NodeOut data, NodeOut pred, const GraphDefBuilder::Options&
                opts);

// Forwards `data` to the output port determined by `pred`.
//
// If `pred` is true, the `data` input is forwarded to `output_true`. Otherwise,
// the data goes to `output_false`.
//
// See also `RefSwitch` and `Merge`.
//
// Arguments:
// * data: The tensor to be forwarded to the appropriate output.
// * pred: A scalar that specifies which output port will receive data.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with outputs:
// * output_false: If `pred` is false, data will be forwarded to this output.
// * output_true: If `pred` is true, data will be forwarded to this output.
Node* Switch(NodeOut data, NodeOut pred, const GraphDefBuilder::Options& opts);

}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_OPS_CONTROL_FLOW_OPS_H_
