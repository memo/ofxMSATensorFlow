// This file is MACHINE GENERATED! Do not edit.

#ifndef TENSORFLOW_CC_OPS_STATE_OPS_H_
#define TENSORFLOW_CC_OPS_STATE_OPS_H_

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


// Update 'ref' by assigning 'value' to it.
//
// This operation outputs "ref" after the assignment is done.
// This makes it easier to chain operations that need to use the reset value.
//
// Arguments:
// * ref: Should be from a `Variable` node. May be uninitialized.
// * value: The value to be assigned to the variable.
// * opts:
//   .WithAttr("validate_shape", bool): Defaults to true.
//     If true, the operation will validate that the shape
// of 'value' matches the shape of the Tensor being assigned to.  If false,
// 'ref' will take on the shape of 'value'.
//   .WithAttr("use_locking", bool): Defaults to true.
//     If True, the assignment will be protected by a lock;
// otherwise the behavior is undefined, but may exhibit less contention.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Same as "ref".  Returned as a convenience for operations that want
// to use the new value after the variable has been reset.
Node* Assign(NodeOut ref, NodeOut value, const GraphDefBuilder::Options& opts);

// Update 'ref' by adding 'value' to it.
//
// This operation outputs "ref" after the update is done.
// This makes it easier to chain operations that need to use the reset value.
//
// Arguments:
// * ref: Should be from a `Variable` node.
// * value: The value to be added to the variable.
// * opts:
//   .WithAttr("use_locking", bool): Defaults to false.
//     If True, the addition will be protected by a lock;
// otherwise the behavior is undefined, but may exhibit less contention.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Same as "ref".  Returned as a convenience for operations that want
// to use the new value after the variable has been updated.
Node* AssignAdd(NodeOut ref, NodeOut value, const GraphDefBuilder::Options&
                opts);

// Update 'ref' by subtracting 'value' from it.
//
// This operation outputs "ref" after the update is done.
// This makes it easier to chain operations that need to use the reset value.
//
// Arguments:
// * ref: Should be from a `Variable` node.
// * value: The value to be subtracted to the variable.
// * opts:
//   .WithAttr("use_locking", bool): Defaults to false.
//     If True, the subtraction will be protected by a lock;
// otherwise the behavior is undefined, but may exhibit less contention.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Same as "ref".  Returned as a convenience for operations that want
// to use the new value after the variable has been updated.
Node* AssignSub(NodeOut ref, NodeOut value, const GraphDefBuilder::Options&
                opts);

// Increments 'ref' until it reaches 'limit'.
//
// This operation outputs "ref" after the update is done.  This makes it
// easier to chain operations that need to use the updated value.
//
// Arguments:
// * ref: Should be from a scalar `Variable` node.
// * limit: If incrementing ref would bring it above limit, instead generates an
// 'OutOfRange' error.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// A copy of the input before increment. If nothing else modifies the
// input, the values produced will all be distinct.
Node* CountUpTo(NodeOut ref, int64 limit, const GraphDefBuilder::Options&
                opts);

// Destroys the temporary variable and returns its final value.
//
// Sets output to the value of the Tensor pointed to by 'ref', then destroys
// the temporary variable called 'var_name'.
// All other uses of 'ref' *must* have executed before this op.
// This is typically achieved by chaining the ref through each assign op, or by
// using control dependencies.
//
// Outputs the final value of the tensor pointed to by 'ref'.
//
// Arguments:
// * ref: A reference to the temporary variable tensor.
// * var_name: Name of the temporary variable, usually the name of the matching
// 'TemporaryVariable' op.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* DestroyTemporaryVariable(NodeOut ref, StringPiece var_name, const
                               GraphDefBuilder::Options& opts);

// Adds sparse updates to a variable reference.
//
// This operation computes
//
//     # Scalar indices
//     ref[indices, ...] += updates[...]
//
//     # Vector indices (for each i)
//     ref[indices[i], ...] += updates[i, ...]
//
//     # High rank indices (for each i, ..., j)
//     ref[indices[i, ..., j], ...] += updates[i, ..., j, ...]
//
// This operation outputs `ref` after the update is done.
// This makes it easier to chain operations that need to use the reset value.
//
// Duplicate entries are handled correctly: if multiple `indices` reference
// the same location, their contributions add.
//
// Requires `updates.shape = indices.shape + ref.shape[1:]`.
//
// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
// <img style="width:100%" src="../../images/ScatterAdd.png" alt>
// </div>
//
// Arguments:
// * ref: Should be from a `Variable` node.
// * indices: A tensor of indices into the first dimension of `ref`.
// * updates: A tensor of updated values to add to `ref`.
// * opts:
//   .WithAttr("use_locking", bool): Defaults to false.
//     If True, the addition will be protected by a lock;
// otherwise the behavior is undefined, but may exhibit less contention.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Same as `ref`.  Returned as a convenience for operations that want
// to use the updated values after the update is done.
Node* ScatterAdd(NodeOut ref, NodeOut indices, NodeOut updates, const
                 GraphDefBuilder::Options& opts);

// Subtracts sparse updates to a variable reference.
//
//     # Scalar indices
//     ref[indices, ...] -= updates[...]
//
//     # Vector indices (for each i)
//     ref[indices[i], ...] -= updates[i, ...]
//
//     # High rank indices (for each i, ..., j)
//     ref[indices[i, ..., j], ...] -= updates[i, ..., j, ...]
//
// This operation outputs `ref` after the update is done.
// This makes it easier to chain operations that need to use the reset value.
//
// Duplicate entries are handled correctly: if multiple `indices` reference
// the same location, their (negated) contributions add.
//
// Requires `updates.shape = indices.shape + ref.shape[1:]`.
//
// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
// <img style="width:100%" src="../../images/ScatterSub.png" alt>
// </div>
//
// Arguments:
// * ref: Should be from a `Variable` node.
// * indices: A tensor of indices into the first dimension of `ref`.
// * updates: A tensor of updated values to subtract from `ref`.
// * opts:
//   .WithAttr("use_locking", bool): Defaults to false.
//     If True, the subtraction will be protected by a lock;
// otherwise the behavior is undefined, but may exhibit less contention.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Same as `ref`.  Returned as a convenience for operations that want
// to use the updated values after the update is done.
Node* ScatterSub(NodeOut ref, NodeOut indices, NodeOut updates, const
                 GraphDefBuilder::Options& opts);

// Applies sparse updates to a variable reference.
//
// This operation computes
//
//     # Scalar indices
//     ref[indices, ...] = updates[...]
//
//     # Vector indices (for each i)
//     ref[indices[i], ...] = updates[i, ...]
//
//     # High rank indices (for each i, ..., j)
//     ref[indices[i, ..., j], ...] = updates[i, ..., j, ...]
//
// This operation outputs `ref` after the update is done.
// This makes it easier to chain operations that need to use the reset value.
//
// If values in `ref` is to be updated more than once, because there are
// duplicate entires in `indices`, the order at which the updates happen
// for each value is undefined.
//
// Requires `updates.shape = indices.shape + ref.shape[1:]`.
//
// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
// <img style="width:100%" src="../../images/ScatterUpdate.png" alt>
// </div>
//
// Arguments:
// * ref: Should be from a `Variable` node.
// * indices: A tensor of indices into the first dimension of `ref`.
// * updates: A tensor of updated values to store in `ref`.
// * opts:
//   .WithAttr("use_locking", bool): Defaults to true.
//     If True, the assignment will be protected by a lock;
// otherwise the behavior is undefined, but may exhibit less contention.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Same as `ref`.  Returned as a convenience for operations that want
// to use the updated values after the update is done.
Node* ScatterUpdate(NodeOut ref, NodeOut indices, NodeOut updates, const
                    GraphDefBuilder::Options& opts);

// Returns a tensor that may be mutated, but only persists within a single step.
//
// This is an experimental op for internal use only and it is possible to use this
// op in unsafe ways.  DO NOT USE unless you fully understand the risks.
//
// It is the caller's responsibility to ensure that 'ref' is eventually passed to a
// matching 'DestroyTemporaryVariable' op after all other uses have completed.
//
// Outputs a ref to the tensor state so it may be read or modified.
//
//   E.g.
//       var = state_ops._temporary_variable([1, 2], types.float_)
//       var_name = var.op.name
//       var = state_ops.assign(var, [[4.0, 5.0]])
//       var = state_ops.assign_add(var, [[6.0, 7.0]])
//       final = state_ops._destroy_temporary_variable(var, var_name=var_name)
//
// Arguments:
// * shape: The shape of the variable tensor.
// * dtype: The type of elements in the variable tensor.
// * opts:
//   .WithAttr("var_name", StringPiece): Defaults to "".
//     Overrides the name used for the temporary variable resource. Default
// value is the name of the 'TemporaryVariable' op (which is guaranteed unique).
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// A reference to the variable tensor.
Node* TemporaryVariable(TensorShape shape, DataType dtype, const
                        GraphDefBuilder::Options& opts);

// Holds state in the form of a tensor that persists across steps.
//
// Outputs a ref to the tensor state so it may be read or modified.
// TODO(zhifengc/mrry): Adds a pointer to a more detail document
// about sharing states in tensorflow.
//
// Arguments:
// * shape: The shape of the variable tensor.
// * dtype: The type of elements in the variable tensor.
// * opts:
//   .WithAttr("container", StringPiece): Defaults to "".
//     If non-empty, this variable is placed in the given container.
// Otherwise, a default container is used.
//   .WithAttr("shared_name", StringPiece): Defaults to "".
//     If non-empty, this variable is named in the given bucket
// with this shared_name. Otherwise, the node name is used instead.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// A reference to the variable tensor.
Node* Variable(TensorShape shape, DataType dtype, const
               GraphDefBuilder::Options& opts);

}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_OPS_STATE_OPS_H_
