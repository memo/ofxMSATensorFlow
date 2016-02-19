// This file is MACHINE GENERATED! Do not edit.

#ifndef TENSORFLOW_CC_OPS_DATA_FLOW_OPS_H_
#define TENSORFLOW_CC_OPS_DATA_FLOW_OPS_H_

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


// Partitions `data` into `num_partitions` tensors using indices from `partitions`.
//
// For each index tuple `js` of size `partitions.ndim`, the slice `data[js, ...]`
// becomes part of `outputs[partitions[js]]`.  The slices with `partitions[js] = i`
// are placed in `outputs[i]` in lexicographic order of `js`, and the first
// dimension of `outputs[i]` is the number of entries in `partitions` equal to `i`.
// In detail,
//
//     outputs[i].shape = [sum(partitions == i)] + data.shape[partitions.ndim:]
//
//     outputs[i] = pack([data[js, ...] for js if partitions[js] == i])
//
// `data.shape` must start with `partitions.shape`.
//
// For example:
//
//     # Scalar partitions
//     partitions = 1
//     num_partitions = 2
//     data = [10, 20]
//     outputs[0] = []  # Empty with shape [0, 2]
//     outputs[1] = [[10, 20]]
//
//     # Vector partitions
//     partitions = [0, 0, 1, 1, 0]
//     num_partitions = 2
//     data = [10, 20, 30, 40, 50]
//     outputs[0] = [10, 20, 50]
//     outputs[1] = [30, 40]
//
// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
// <img style="width:100%" src="../../images/DynamicPartition.png" alt>
// </div>
//
// Arguments:
// * partitions: Any shape.  Indices in the range `[0, num_partitions)`.
// * num_partitions: The number of partitions to output.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* DynamicPartition(NodeOut data, NodeOut partitions, int64 num_partitions,
                       const GraphDefBuilder::Options& opts);

// Interleave the values from the `data` tensors into a single tensor.
//
// Builds a merged tensor such that
//
//     merged[indices[m][i, ..., j], ...] = data[m][i, ..., j, ...]
//
// For example, if each `indices[m]` is scalar or vector, we have
//
//     # Scalar indices
//     merged[indices[m], ...] = data[m][...]
//
//     # Vector indices
//     merged[indices[m][i], ...] = data[m][i, ...]
//
// Each `data[i].shape` must start with the corresponding `indices[i].shape`,
// and the rest of `data[i].shape` must be constant w.r.t. `i`.  That is, we
// must have `data[i].shape = indices[i].shape + constant`.  In terms of this
// `constant`, the output shape is
//
//     merged.shape = [max(indices)] + constant
//
// Values are merged in order, so if an index appears in both `indices[m][i]` and
// `indices[n][j]` for `(m,i) < (n,j)` the slice `data[n][j]` will appear in the
// merged result.
//
// For example:
//
//     indices[0] = 6
//     indices[1] = [4, 1]
//     indices[2] = [[5, 2], [0, 3]]
//     data[0] = [61, 62]
//     data[1] = [[41, 42], [11, 12]]
//     data[2] = [[[51, 52], [21, 22]], [[1, 2], [31, 32]]]
//     merged = [[1, 2], [11, 12], [21, 22], [31, 32], [41, 42],
//               [51, 52], [61, 62]]
//
// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
// <img style="width:100%" src="../../images/DynamicStitch.png" alt>
// </div>
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* DynamicStitch(gtl::ArraySlice<NodeOut> indices, gtl::ArraySlice<NodeOut>
                    data, const GraphDefBuilder::Options& opts);

// A queue that produces elements in first-in first-out order.
//
// Arguments:
// * component_types: The type of each component in a value.
// * opts:
//   .WithAttr("shapes", gtl::ArraySlice<TensorShape>): Defaults to [].
//     The shape of each component in a value. The length of this attr must
// be either 0 or the same as the length of component_types. If the length of
// this attr is 0, the shapes of queue elements are not constrained, and
// only one element may be dequeued at a time.
//   .WithAttr("capacity", int64): Defaults to -1.
//     The upper bound on the number of elements in this queue.
// Negative numbers mean no limit.
//   .WithAttr("container", StringPiece): Defaults to "".
//     If non-empty, this queue is placed in the given container.
// Otherwise, a default container is used.
//   .WithAttr("shared_name", StringPiece): Defaults to "".
//     If non-empty, this queue will be shared under the given name
// across multiple sessions.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The handle to the queue.
Node* FIFOQueue(DataTypeSlice component_types, const GraphDefBuilder::Options&
                opts);

// Creates a non-initialized hash table.
//
// This op creates a hash table, specifying the type of its keys and values.
// Before using the table you will have to initialize it.  After initialization the
// table will be immutable.
//
// Arguments:
// * key_dtype: Type of the table keys.
// * value_dtype: Type of the table values.
// * opts:
//   .WithAttr("container", StringPiece): Defaults to "".
//     If non-empty, this table is placed in the given container.
// Otherwise, a default container is used.
//   .WithAttr("shared_name", StringPiece): Defaults to "".
//     If non-empty, this table is shared under the given name across
// multiple sessions.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Handle to a table.
Node* HashTable(DataType key_dtype, DataType value_dtype, const
                GraphDefBuilder::Options& opts);

// Table initializer that takes two tensors for keys and values respectively.
//
// Arguments:
// * table_handle: Handle to a table which will be initialized.
// * keys: Keys of type Tkey.
// * values: Values of type Tval. Same shape as `keys`.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* InitializeTable(NodeOut table_handle, NodeOut keys, NodeOut values, const
                      GraphDefBuilder::Options& opts);

// Looks up keys in a table, outputs the corresponding values.
//
// The tensor `keys` must of the same type as the keys of the table.
// The output `values` is of the type of the table values.
//
// The scalar `default_value` is the value output for keys not present in the
// table. It must also be of the same type as the table values.
//
// Arguments:
// * table_handle: Handle to the table.
// * keys: Any shape.  Keys to look up.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Same shape as `keys`.  Values found in the table, or `default_values`
// for missing keys.
Node* LookupTableFind(NodeOut table_handle, NodeOut keys, NodeOut
                      default_value, const GraphDefBuilder::Options& opts);

// Computes the number of elements in the given table.
//
// Arguments:
// * table_handle: Handle to the table.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Scalar that contains number of elements in the table.
Node* LookupTableSize(NodeOut table_handle, const GraphDefBuilder::Options&
                      opts);

// A queue that produces elements in first-in first-out order.
//
// Variable-size shapes are allowed by setting the corresponding shape dimensions
// to 0 in the shape attr.  In this case DequeueMany will pad up to the maximum
// size of any given element in the minibatch.  See below for details.
//
// Arguments:
// * component_types: The type of each component in a value.
// * opts:
//   .WithAttr("shapes", gtl::ArraySlice<TensorShape>): Defaults to [].
//     The shape of each component in a value. The length of this attr must
// be either 0 or the same as the length of component_types.
// Shapes of fixed rank but variable size are allowed by setting
// any shape dimension to -1.  In this case, the inputs' shape may vary along
// the given dimension, and DequeueMany will pad the given dimension with
// zeros up to the maximum shape of all elements in the given batch.
// If the length of this attr is 0, different queue elements may have
// different ranks and shapes, but only one element may be dequeued at a time.
//   .WithAttr("capacity", int64): Defaults to -1.
//     The upper bound on the number of elements in this queue.
// Negative numbers mean no limit.
//   .WithAttr("container", StringPiece): Defaults to "".
//     If non-empty, this queue is placed in the given container.
// Otherwise, a default container is used.
//   .WithAttr("shared_name", StringPiece): Defaults to "".
//     If non-empty, this queue will be shared under the given name
// across multiple sessions.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The handle to the queue.
Node* PaddingFIFOQueue(DataTypeSlice component_types, const
                       GraphDefBuilder::Options& opts);

// Closes the given queue.
//
// This operation signals that no more elements will be enqueued in the
// given queue. Subsequent Enqueue(Many) operations will fail.
// Subsequent Dequeue(Many) operations will continue to succeed if
// sufficient elements remain in the queue. Subsequent Dequeue(Many)
// operations that would block will fail immediately.
//
// Arguments:
// * handle: The handle to a queue.
// * opts:
//   .WithAttr("cancel_pending_enqueues", bool): Defaults to false.
//     If true, all pending enqueue requests that are
// blocked on the given queue will be cancelled.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* QueueClose(NodeOut handle, const GraphDefBuilder::Options& opts);

// Dequeues a tuple of one or more tensors from the given queue.
//
// This operation has k outputs, where k is the number of components
// in the tuples stored in the given queue, and output i is the ith
// component of the dequeued tuple.
//
// N.B. If the queue is empty, this operation will block until an element
// has been dequeued (or 'timeout_ms' elapses, if specified).
//
// Arguments:
// * handle: The handle to a queue.
// * component_types: The type of each component in a tuple.
// * opts:
//   .WithAttr("timeout_ms", int64): Defaults to -1.
//     If the queue is empty, this operation will block for up to
// timeout_ms milliseconds.
// Note: This option is not supported yet.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// One or more tensors that were dequeued as a tuple.
Node* QueueDequeue(NodeOut handle, DataTypeSlice component_types, const
                   GraphDefBuilder::Options& opts);

// Dequeues n tuples of one or more tensors from the given queue.
//
// This operation concatenates queue-element component tensors along the
// 0th dimension to make a single component tensor.  All of the components
// in the dequeued tuple will have size n in the 0th dimension.
//
// This operation has k outputs, where k is the number of components in
// the tuples stored in the given queue, and output i is the ith
// component of the dequeued tuple.
//
// N.B. If the queue is empty, this operation will block until n elements
// have been dequeued (or 'timeout_ms' elapses, if specified).
//
// Arguments:
// * handle: The handle to a queue.
// * n: The number of tuples to dequeue.
// * component_types: The type of each component in a tuple.
// * opts:
//   .WithAttr("timeout_ms", int64): Defaults to -1.
//     If the queue has fewer than n elements, this operation
// will block for up to timeout_ms milliseconds.
// Note: This option is not supported yet.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// One or more tensors that were dequeued as a tuple.
Node* QueueDequeueMany(NodeOut handle, NodeOut n, DataTypeSlice
                       component_types, const GraphDefBuilder::Options& opts);

// Enqueues a tuple of one or more tensors in the given queue.
//
// The components input has k elements, which correspond to the components of
// tuples stored in the given queue.
//
// N.B. If the queue is full, this operation will block until the given
// element has been enqueued (or 'timeout_ms' elapses, if specified).
//
// Arguments:
// * handle: The handle to a queue.
// * components: One or more tensors from which the enqueued tensors should be taken.
// * opts:
//   .WithAttr("timeout_ms", int64): Defaults to -1.
//     If the queue is full, this operation will block for up to
// timeout_ms milliseconds.
// Note: This option is not supported yet.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* QueueEnqueue(NodeOut handle, gtl::ArraySlice<NodeOut> components, const
                   GraphDefBuilder::Options& opts);

// Enqueues zero or more tuples of one or more tensors in the given queue.
//
// This operation slices each component tensor along the 0th dimension to
// make multiple queue elements. All of the tuple components must have the
// same size in the 0th dimension.
//
// The components input has k elements, which correspond to the components of
// tuples stored in the given queue.
//
// N.B. If the queue is full, this operation will block until the given
// elements have been enqueued (or 'timeout_ms' elapses, if specified).
//
// Arguments:
// * handle: The handle to a queue.
// * components: One or more tensors from which the enqueued tensors should
// be taken.
// * opts:
//   .WithAttr("timeout_ms", int64): Defaults to -1.
//     If the queue is too full, this operation will block for up
// to timeout_ms milliseconds.
// Note: This option is not supported yet.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* QueueEnqueueMany(NodeOut handle, gtl::ArraySlice<NodeOut> components,
                       const GraphDefBuilder::Options& opts);

// Computes the number of elements in the given queue.
//
// Arguments:
// * handle: The handle to a queue.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The number of elements in the given queue.
Node* QueueSize(NodeOut handle, const GraphDefBuilder::Options& opts);

// A queue that randomizes the order of elements.
//
// Arguments:
// * component_types: The type of each component in a value.
// * opts:
//   .WithAttr("shapes", gtl::ArraySlice<TensorShape>): Defaults to [].
//     The shape of each component in a value. The length of this attr must
// be either 0 or the same as the length of component_types. If the length of
// this attr is 0, the shapes of queue elements are not constrained, and
// only one element may be dequeued at a time.
//   .WithAttr("capacity", int64): Defaults to -1.
//     The upper bound on the number of elements in this queue.
// Negative numbers mean no limit.
//   .WithAttr("min_after_dequeue", int64): Defaults to 0.
//     Dequeue will block unless there would be this
// many elements after the dequeue or the queue is closed. This
// ensures a minimum level of mixing of elements.
//   .WithAttr("seed", int64): Defaults to 0.
//     If either seed or seed2 is set to be non-zero, the random number
// generator is seeded by the given seed.  Otherwise, a random seed is used.
//   .WithAttr("seed2", int64): Defaults to 0.
//     A second seed to avoid seed collision.
//   .WithAttr("container", StringPiece): Defaults to "".
//     If non-empty, this queue is placed in the given container.
// Otherwise, a default container is used.
//   .WithAttr("shared_name", StringPiece): Defaults to "".
//     If non-empty, this queue will be shared under the given name
// across multiple sessions.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The handle to the queue.
Node* RandomShuffleQueue(DataTypeSlice component_types, const
                         GraphDefBuilder::Options& opts);

// A stack that produces elements in first-in last-out order.
//
// Arguments:
// * elem_type: The type of the elements on the stack.
// * opts:
//   .WithAttr("stack_name", StringPiece): Defaults to "".
//     Overrides the name used for the temporary stack resource. Default
// value is the name of the 'Stack' op (which is guaranteed unique).
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The handle to the stack.
Node* Stack(DataType elem_type, const GraphDefBuilder::Options& opts);

// Delete the stack from its resource container.
//
// Arguments:
// * handle: The handle to a stack.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* StackClose(NodeOut handle, const GraphDefBuilder::Options& opts);

// Pop the element at the top of the stack.
//
// Arguments:
// * handle: The handle to a stack.
// * elem_type: The type of the elem that is popped.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The tensor that is popped from the top of the stack.
Node* StackPop(NodeOut handle, DataType elem_type, const
               GraphDefBuilder::Options& opts);

// Push an element onto the stack.
//
// Arguments:
// * handle: The handle to a stack.
// * elem: The tensor to be pushed onto the stack.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The same tensor as the input 'elem'.
Node* StackPush(NodeOut handle, NodeOut elem, const GraphDefBuilder::Options&
                opts);

// An array of Tensors of given size, with data written via Write and read
//
// via Read or Pack.
//
// Arguments:
// * size: The size of the array.
// * dtype: The type of the elements on the tensor_array.
// * opts:
//   .WithAttr("dynamic_size", bool): Defaults to false.
//     A boolean that determines whether writes to the TensorArray
// are allowed to grow the size.  By default, this is not allowed.
//   .WithAttr("tensor_array_name", StringPiece): Defaults to "".
//     Overrides the name used for the temporary tensor_array
// resource. Default value is the name of the 'TensorArray' op (which
// is guaranteed unique).
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The handle to the TensorArray.
Node* TensorArray(NodeOut size, DataType dtype, const GraphDefBuilder::Options&
                  opts);

// Delete the TensorArray from its resource container.  This enables
//
// the user to close and release the resource in the middle of a step/run.
//
// Arguments:
// * handle: The handle to a TensorArray (output of TensorArray or TensorArrayGrad).
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* TensorArrayClose(NodeOut handle, const GraphDefBuilder::Options& opts);

// Concat the elements from the TensorArray.
//
// Takes T elements of shapes (n0 x d0 x d1 x ...), (n1 x d0 x d1 x ...),
//   ..., (n(T-1) x d0 x d1 x ...)
// and concatenates them into a Tensor of shape:
//   (n0 + n1 + ... + n(T-1) x d0 x d1 x ...).
//
// All elements must have the same shape (excepting the first dimension).
//
// Arguments:
// * handle: The handle to a TensorArray.
// * flow_in: A float scalar that enforces proper chaining of operations.
// * dtype: The type of the elem that is returned.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with outputs:
// * value: All of the elements in the TensorArray, concatenated along the first
// axis.
// * lengths: A vector of the row sizes of the original T elements in the
// value output.  In the example above, this would be the values:
// (n1, n2, ..., n(T-1))
Node* TensorArrayConcat(NodeOut handle, NodeOut flow_in, DataType dtype, const
                        GraphDefBuilder::Options& opts);

// Creates a TensorArray for storing the gradients of values in the given handle.
//
// If the given TensorArray gradient already exists, returns a reference to it.
//
// Locks the size of the original TensorArray by disabling its dynamic size flag.
//
// **A note about the input flow_in:**
//
// The handle flow_in forces the execution of the gradient lookup to occur
// only after certain other operations have occurred.  For example, when
// the forward TensorArray is dynamically sized, writes to this TensorArray
// may resize the object.  The gradient TensorArray is statically sized based
// on the size of the forward TensorArray when this operation executes.
// Furthermore, the size of the forward TensorArray is frozen by this call.
// As a result, the flow is used to ensure that the call to generate the gradient
// TensorArray only happens after all writes are executed.
//
// In terms of e.g. python TensorArray sugar wrappers when using dynamically sized
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* TensorArrayGrad(NodeOut handle, NodeOut flow_in, StringPiece source,
                      const GraphDefBuilder::Options& opts);

// Pack the elements from the TensorArray.
//
// All elements must have the same shape.
//
// Arguments:
// * handle: The handle to a TensorArray.
// * flow_in: A float scalar that enforces proper chaining of operations.
// * dtype: The type of the elem that is returned.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// All of the elements in the TensorArray, concatenated along a new
// axis (the new dimension 0).
Node* TensorArrayPack(NodeOut handle, NodeOut flow_in, DataType dtype, const
                      GraphDefBuilder::Options& opts);

// Read an element from the TensorArray.
//
// Arguments:
// * handle: The handle to a TensorArray.
// * flow_in: A float scalar that enforces proper chaining of operations.
// * dtype: The type of the elem that is returned.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The tensor that is read from the TensorArray.
Node* TensorArrayRead(NodeOut handle, NodeOut index, NodeOut flow_in, DataType
                      dtype, const GraphDefBuilder::Options& opts);

// Get the current size of the TensorArray.
//
// Arguments:
// * handle: The handle to a TensorArray (output of TensorArray or TensorArrayGrad).
// * flow_in: A float scalar that enforces proper chaining of operations.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The current size of the TensorArray.
Node* TensorArraySize(NodeOut handle, NodeOut flow_in, const
                      GraphDefBuilder::Options& opts);

// Split the data from the input value into TensorArray elements.
//
// Assuming that `lengths` takes on values
//   (n0, n1, ..., n(T-1))
// and that `value` has shape
//   (n0 + n1 + ... + n(T-1) x d0 x d1 x ...),
// this splits values into a TensorArray with T tensors.
//
// TensorArray index t will be the subtensor of values with starting position
//   (n0 + n1 + ... + n(t-1), 0, 0, ...)
// and having size
//   nt x d0 x d1 x ...
//
// Arguments:
// * handle: The handle to a TensorArray.
// * value: The concatenated tensor to write to the TensorArray.
// * lengths: The vector of lengths, how to split the rows of value into the
// TensorArray.
// * flow_in: A float scalar that enforces proper chaining of operations.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// A float scalar that enforces proper chaining of operations.
Node* TensorArraySplit(NodeOut handle, NodeOut value, NodeOut lengths, NodeOut
                       flow_in, const GraphDefBuilder::Options& opts);

// Unpack the data from the input value into TensorArray elements.
//
// Arguments:
// * handle: The handle to a TensorArray.
// * value: The concatenated tensor to write to the TensorArray.
// * flow_in: A float scalar that enforces proper chaining of operations.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// A float scalar that enforces proper chaining of operations.
Node* TensorArrayUnpack(NodeOut handle, NodeOut value, NodeOut flow_in, const
                        GraphDefBuilder::Options& opts);

// Push an element onto the tensor_array.
//
// Arguments:
// * handle: The handle to a TensorArray.
// * index: The position to write to inside the TensorArray.
// * value: The tensor to write to the TensorArray.
// * flow_in: A float scalar that enforces proper chaining of operations.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// A float scalar that enforces proper chaining of operations.
Node* TensorArrayWrite(NodeOut handle, NodeOut index, NodeOut value, NodeOut
                       flow_in, const GraphDefBuilder::Options& opts);

}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_OPS_DATA_FLOW_OPS_H_
