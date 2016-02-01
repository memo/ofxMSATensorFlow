// This file is MACHINE GENERATED! Do not edit.

#ifndef TENSORFLOW_CC_OPS_ARRAY_OPS_H_
#define TENSORFLOW_CC_OPS_ARRAY_OPS_H_

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


// Return the reduction indices for computing gradients of s0 op s1 with broadcast.
//
// This is typically used by gradient computations for a broadcasting operation.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with outputs:
// * r0
// * r1
Node* BroadcastGradientArgs(NodeOut s0, NodeOut s1, const
                            GraphDefBuilder::Options& opts);

// Checks a tensor for NaN and Inf values.
//
// When run, reports an `InvalidArgument` error if `tensor` has any values
// that are not a number (NaN) or infinity (Inf). Otherwise, passes `tensor` as-is.
//
// Arguments:
// * message: Prefix of the error message.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* CheckNumerics(NodeOut tensor, StringPiece message, const
                    GraphDefBuilder::Options& opts);

// Concatenates tensors along one dimension.
//
// Arguments:
// * concat_dim: 0-D.  The dimension along which to concatenate.  Must be in the
// range [0, rank(values)).
// * values: The `N` Tensors to concatenate. Their ranks and types must match,
// and their sizes must match in all dimensions except `concat_dim`.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// A `Tensor` with the concatenation of values stacked along the
// `concat_dim` dimension.  This tensor's shape matches that of `values` except
// in `concat_dim` where it has the sum of the sizes.
Node* Concat(NodeOut concat_dim, gtl::ArraySlice<NodeOut> values, const
             GraphDefBuilder::Options& opts);

// Computes offsets of concat inputs within its output.
//
// For example:
//
// ```prettyprint
// # 'x' is [2, 2, 7]
// # 'y' is [2, 3, 7]
// # 'z' is [2, 5, 7]
// concat_offset(2, [x, y, z]) => [0, 0, 0], [0, 2, 0], [0, 5, 0]
// ```
//
// Arguments:
// * concat_dim: The dimension along which to concatenate.
// * shape: The `N` int32 vetors representing shape of tensors being concatenated.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* ConcatOffset(NodeOut concat_dim, gtl::ArraySlice<NodeOut> shape, const
                   GraphDefBuilder::Options& opts);

// Returns a constant tensor.
//
// Arguments:
// * value: Attr `value` is the tensor to return.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* Const(const Tensor& value, DataType dtype, const
            GraphDefBuilder::Options& opts);

// DepthToSpace for tensors of type T.
//
// Rearranges data from depth into blocks of spatial data.
// This is the reverse transformation of SpaceToDepth. More specifically,
// this op outputs a copy of the input tensor where values from the `depth`
// dimension are moved in spatial blocks to the `height` and `width` dimensions.
// The attr `block_size` indicates the input block size and how the data is moved.
//
//   * Chunks of data of size `block_size * block_size` from depth are rearranged
//     into non-overlapping blocks of size `block_size x block_size`
//   * The width the output tensor is `input_depth * block_size`, whereas the
//     height is `input_height * block_size`.
//   * The depth of the input tensor must be divisible by
//     `block_size * block_size`.
//
// That is, assuming the input is in the shape:
// `[batch, height, width, depth]`,
// the shape of the output will be:
// `[batch, height*block_size, width*block_size, depth/(block_size*block_size)]`
//
// This operation requires that the input tensor be of rank 4, and that
// `block_size` be >=1 and that `block_size * block_size` be a divisor of the
// input depth.
//
// This operation is useful for resizing the activations between convolutions
// (but keeping all data), e.g. instead of pooling. It is also useful for training
// purely convolutional models.
//
// For example, given this input of shape `[1, 1, 1, 4]`, and a block size of 2:
//
// ```prettyprint
// x = [[[[1, 2, 3, 4]]]]
//
// ```
//
// This operation will output a tensor of shape `[1, 2, 2, 1]`:
//
// ```prettyprint
//    [[[[1], [2]],
//      [[3], [4]]]]
// ```
//
// Here, the input has a batch of 1 and each batch element has shape `[1, 1, 4]`,
// the corresponding output will have 2x2 elements and will have a depth of
// 1 channel (1 = `4 / (block_size * block_size)`).
// The output element shape is `[2, 2, 1]`.
//
// For an input tensor with larger depth, here of shape `[1, 1, 1, 12]`, e.g.
//
// ```prettyprint
// x = [[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]]]
// ```
//
// This operation, for block size of 2, will return the following tensor of shape
// `[1, 2, 2, 3]`
//
// ```prettyprint
//    [[[[1, 2, 3], [4, 5, 6]],
//      [[7, 8, 9], [10, 11, 12]]]]
//
// ```
//
// Similarly, for the following input of shape `[1 2 2 4]`, and a block size of 2:
//
// ```prettyprint
// x =  [[[[1, 2, 3, 4],
//        [5, 6, 7, 8]],
//       [[9, 10, 11, 12],
//        [13, 14, 15, 16]]]]
// ```
//
// the operator will return the following tensor of shape `[1 4 4 1]`:
//
// ```prettyprint
// x = [[ [1],   [2],  [5],  [6]],
//      [ [3],   [4],  [7],  [8]],
//      [ [9],  [10], [13],  [14]],
//      [ [11], [12], [15],  [16]]]
//
// ```
//
// Arguments:
// * block_size: The size of the spatial block, same as in Space2Depth.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* DepthToSpace(NodeOut input, int64 block_size, const
                   GraphDefBuilder::Options& opts);

// Returns a diagonal tensor with a given diagonal values.
//
// Given a `diagonal`, this operation returns a tensor with the `diagonal` and
// everything else padded with zeros. The diagonal is computed as follows:
//
// Assume `diagonal` has dimensions [D1,..., Dk], then the output is a tensor of
// rank 2k with dimensions [D1,..., Dk, D1,..., Dk] where:
//
// `output[i1,..., ik, i1,..., ik] = diagonal[i1, ..., ik]` and 0 everywhere else.
//
// For example:
//
// ```prettyprint
// # 'diagonal' is [1, 2, 3, 4]
// tf.diag(diagonal) ==> [[1, 0, 0, 0]
//                        [0, 2, 0, 0]
//                        [0, 0, 3, 0]
//                        [0, 0, 0, 4]]
// ```
//
// Arguments:
// * diagonal: Rank k tensor where k is at most 3.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* Diag(NodeOut diagonal, const GraphDefBuilder::Options& opts);

// Computes the (possibly normalized) Levenshtein Edit Distance.
//
// The inputs are variable-length sequences provided by SparseTensors
//   (hypothesis_indices, hypothesis_values, hypothesis_shape)
// and
//   (truth_indices, truth_values, truth_shape).
//
// The inputs are:
//
// Arguments:
// * hypothesis_indices: The indices of the hypothesis list SparseTensor.
// This is an N x R int64 matrix.
// * hypothesis_values: The values of the hypothesis list SparseTensor.
// This is an N-length vector.
// * hypothesis_shape: The shape of the hypothesis list SparseTensor.
// This is an R-length vector.
// * truth_indices: The indices of the truth list SparseTensor.
// This is an M x R int64 matrix.
// * truth_values: The values of the truth list SparseTensor.
// This is an M-length vector.
// * truth_shape: truth indices, vector.
// * opts:
//   .WithAttr("normalize", bool): Defaults to true.
//     boolean (if true, edit distances are normalized by length of truth).
//
// The output is:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// A dense float tensor with rank R - 1.
//
// For the example input:
//
//     // hypothesis represents a 2x1 matrix with variable-length values:
//     //   (0,0) = ["a"]
//     //   (1,0) = ["b"]
//     hypothesis_indices = [[0, 0, 0],
//                           [1, 0, 0]]
//     hypothesis_values = ["a", "b"]
//     hypothesis_shape = [2, 1, 1]
//
//     // truth represents a 2x2 matrix with variable-length values:
//     //   (0,0) = []
//     //   (0,1) = ["a"]
//     //   (1,0) = ["b", "c"]
//     //   (1,1) = ["a"]
//     truth_indices = [[0, 1, 0],
//                      [1, 0, 0],
//                      [1, 0, 1],
//                      [1, 1, 0]]
//     truth_values = ["a", "b", "c", "a"]
//     truth_shape = [2, 2, 2]
//     normalize = true
//
// The output will be:
//
//     // output is a 2x2 matrix with edit distances normalized by truth lengths.
//     output = [[inf, 1.0],  // (0,0): no truth, (0,1): no hypothesis
//               [0.5, 1.0]]  // (1,0): addition, (1,1): no hypothesis
Node* EditDistance(NodeOut hypothesis_indices, NodeOut hypothesis_values,
                   NodeOut hypothesis_shape, NodeOut truth_indices, NodeOut
                   truth_values, NodeOut truth_shape, const
                   GraphDefBuilder::Options& opts);

// Inserts a dimension of 1 into a tensor's shape.
//
// Given a tensor `input`, this operation inserts a dimension of 1 at the
// dimension index `dim` of `input`'s shape. The dimension index `dim` starts at
// zero; if you specify a negative number for `dim` it is counted backward from
// the end.
//
// This operation is useful if you want to add a batch dimension to a single
// element. For example, if you have a single image of shape `[height, width,
// channels]`, you can make it a batch of 1 image with `expand_dims(image, 0)`,
// which will make the shape `[1, height, width, channels]`.
//
// Other examples:
//
// ```prettyprint
// # 't' is a tensor of shape [2]
// shape(expand_dims(t, 0)) ==> [1, 2]
// shape(expand_dims(t, 1)) ==> [2, 1]
// shape(expand_dims(t, -1)) ==> [2, 1]
//
// # 't2' is a tensor of shape [2, 3, 5]
// shape(expand_dims(t2, 0)) ==> [1, 2, 3, 5]
// shape(expand_dims(t2, 2)) ==> [2, 3, 1, 5]
// shape(expand_dims(t2, 3)) ==> [2, 3, 5, 1]
// ```
//
// This operation requires that:
//
// `-1-input.dims() <= dim <= input.dims()`
//
// This operation is related to `squeeze()`, which removes dimensions of
// size 1.
//
// Arguments:
// * dim: 0-D (scalar). Specifies the dimension index at which to
// expand the shape of `input`.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Contains the same data as `input`, but its shape has an additional
// dimension of size 1 added.
Node* ExpandDims(NodeOut input, NodeOut dim, const GraphDefBuilder::Options&
                 opts);

// Creates a tensor filled with a scalar value.
//
// This operation creates a tensor of shape `dims` and fills it with `value`.
//
// For example:
//
// ```prettyprint
// # Output tensor has shape [2, 3].
// fill([2, 3], 9) ==> [[9, 9, 9]
//                      [9, 9, 9]]
// ```
//
// Arguments:
// * dims: 1-D. Represents the shape of the output tensor.
// * value: 0-D (scalar). Value to fill the returned tensor.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* Fill(NodeOut dims, NodeOut value, const GraphDefBuilder::Options& opts);

// Gather slices from `params` according to `indices`.
//
// `indices` must be an integer tensor of any dimension (usually 0-D or 1-D).
// Produces an output tensor with shape `indices.shape + params.shape[1:]` where:
//
//     # Scalar indices
//     output[:, ..., :] = params[indices, :, ... :]
//
//     # Vector indices
//     output[i, :, ..., :] = params[indices[i], :, ... :]
//
//     # Higher rank indices
//     output[i, ..., j, :, ... :] = params[indices[i, ..., j], :, ..., :]
//
// If `indices` is a permutation and `len(indices) == params.shape[0]` then
// this operation will permute `params` accordingly.
//
// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
// <img style="width:100%" src="../../images/Gather.png" alt>
// </div>
//
// Arguments:
// * opts:
//   .WithAttr("validate_indices", bool): Defaults to true.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* Gather(NodeOut params, NodeOut indices, const GraphDefBuilder::Options&
             opts);

// Return a tensor with the same shape and contents as the input tensor or value.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* Identity(NodeOut input, const GraphDefBuilder::Options& opts);

// Computes the inverse permutation of a tensor.
//
// This operation computes the inverse of an index permutation. It takes a 1-D
// integer tensor `x`, which represents the indices of a zero-based array, and
// swaps each value with its index position. In other words, for an output tensor
// `y` and an input tensor `x`, this operation computes the following:
//
// `y[x[i]] = i for i in [0, 1, ..., len(x) - 1]`
//
// The values must include 0. There can be no duplicate values or negative values.
//
// For example:
//
// ```prettyprint
// # tensor `x` is [3, 4, 0, 2, 1]
// invert_permutation(x) ==> [2, 4, 3, 0, 1]
// ```
//
// Arguments:
// * x: 1-D.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// 1-D.
Node* InvertPermutation(NodeOut x, const GraphDefBuilder::Options& opts);

// Computes the difference between two lists of numbers or strings.
//
// Given a list `x` and a list `y`, this operation returns a list `out` that
// represents all values that are in `x` but not in `y`. The returned list `out`
// is sorted in the same order that the numbers appear in `x` (duplicates are
// preserved). This operation also returns a list `idx` that represents the
// position of each `out` element in `x`. In other words:
//
// `out[i] = x[idx[i]] for i in [0, 1, ..., len(out) - 1]`
//
// For example, given this input:
//
// ```prettyprint
// x = [1, 2, 3, 4, 5, 6]
// y = [1, 3, 5]
// ```
//
// This operation would return:
//
// ```prettyprint
// out ==> [2, 4, 6]
// idx ==> [1, 3, 5]
// ```
//
// Arguments:
// * x: 1-D. Values to keep.
// * y: 1-D. Values to remove.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with outputs:
// * out: 1-D. Values present in `x` but not in `y`.
// * idx: 1-D. Positions of `x` values preserved in `out`.
Node* ListDiff(NodeOut x, NodeOut y, const GraphDefBuilder::Options& opts);

// Packs a list of `N` rank-`R` tensors into one rank-`(R+1)` tensor.
//
// Packs the `N` tensors in `values` into a tensor with rank one higher than each
// tensor in `values` and shape `[N] + values[0].shape`. The output satisfies
// `output[i, ...] = values[i][...]`.
//
// This is the opposite of `unpack`.
//
// Arguments:
// * values: Must be of same shape and type.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The packed tensor.
Node* Pack(gtl::ArraySlice<NodeOut> values, const GraphDefBuilder::Options&
           opts);

// Pads a tensor with zeros.
//
// This operation pads a `input` with zeros according to the `paddings` you
// specify. `paddings` is an integer tensor with shape `[Dn, 2]`, where n is the
// rank of `input`. For each dimension D of `input`, `paddings[D, 0]` indicates
// how many zeros to add before the contents of `input` in that dimension, and
// `paddings[D, 1]` indicates how many zeros to add after the contents of `input`
// in that dimension.
//
// The padded size of each dimension D of the output is:
//
// `paddings(D, 0) + input.dim_size(D) + paddings(D, 1)`
//
// For example:
//
// ```prettyprint
// # 't' is [[1, 1], [2, 2]]
// # 'paddings' is [[1, 1], [2, 2]]
// # rank of 't' is 2
// pad(t, paddings) ==> [[0, 0, 0, 0, 0, 0]
//                       [0, 0, 1, 1, 0, 0]
//                       [0, 0, 2, 2, 0, 0]
//                       [0, 0, 0, 0, 0, 0]]
// ```
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* Pad(NodeOut input, NodeOut paddings, const GraphDefBuilder::Options&
          opts);

// A placeholder op for a value that will be fed into the computation.
//
// N.B. This operation will fail with an error if it is executed. It is
// intended as a way to represent a value that will always be fed, and to
// provide attrs that enable the fed value to be checked at runtime.
//
// Arguments:
// * dtype: The type of elements in the tensor.
// * opts:
//   .WithAttr("shape", TensorShape): Defaults to [].
//     (Optional) The shape of the tensor. If the shape has 0 dimensions, the
// shape is unconstrained.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// A placeholder tensor that must be replaced using the feed mechanism.
Node* Placeholder(DataType dtype, const GraphDefBuilder::Options& opts);

// Returns the rank of a tensor.
//
// This operation returns an integer representing the rank of `input`.
//
// For example:
//
// ```prettyprint
// # 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
// # shape of tensor 't' is [2, 2, 3]
// rank(t) ==> 3
// ```
//
// **Note**: The rank of a tensor is not the same as the rank of a matrix. The rank
// of a tensor is the number of indices required to uniquely select each element
// of the tensor. Rank is also known as "order", "degree", or "ndims."
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* Rank(NodeOut input, const GraphDefBuilder::Options& opts);

// Return the same ref tensor as the input ref tensor.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* RefIdentity(NodeOut input, const GraphDefBuilder::Options& opts);

// Reshapes a tensor.
//
// Given `tensor`, this operation returns a tensor that has the same values
// as `tensor` with shape `shape`.
//
// If one component of `shape` is the special value -1, the size of that dimension
// is computed so that the total size remains constant.  In particular, a `shape`
// of `[-1]` flattens into 1-D.  At most one component of `shape` can be -1.
//
// If `shape` is 1-D or higher, then the operation returns a tensor with shape
// `shape` filled with the values of `tensor`. In this case, the number of elements
// implied by `shape` must be the same as the number of elements in `tensor`.
//
// For example:
//
// ```prettyprint
// # tensor 't' is [1, 2, 3, 4, 5, 6, 7, 8, 9]
// # tensor 't' has shape [9]
// reshape(t, [3, 3]) ==> [[1, 2, 3]
//                         [4, 5, 6]
//                         [7, 8, 9]]
//
// # tensor 't' is [[[1, 1], [2, 2]]
// #                [[3, 3], [4, 4]]]
// # tensor 't' has shape [2, 2, 2]
// reshape(t, [2, 4]) ==> [[1, 1, 2, 2]
//                         [3, 3, 4, 4]]
//
// # tensor 't' is [[[1, 1, 1],
// #                 [2, 2, 2]],
// #                [[3, 3, 3],
// #                 [4, 4, 4]],
// #                [[5, 5, 5],
// #                 [6, 6, 6]]]
// # tensor 't' has shape [3, 2, 3]
// # pass '[-1]' to flatten 't'
// reshape(t, [-1]) ==> [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]
// # -1 can also be used with higher dimensional shapes
// reshape(t, [2, -1]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
//                          [4, 4, 4, 5, 5, 5, 6, 6, 6]]
//
// # tensor 't' is [7]
// # shape `[]` reshapes to a scalar
// reshape(t, []) ==> 7
// ```
//
// Arguments:
// * shape: Defines the shape of the output tensor.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* Reshape(NodeOut tensor, NodeOut shape, const GraphDefBuilder::Options&
              opts);

// Reverses specific dimensions of a tensor.
//
// Given a `tensor`, and a `bool` tensor `dims` representing the dimensions
// of `tensor`, this operation reverses each dimension i of `tensor` where
// `dims[i]` is `True`.
//
// `tensor` can have up to 8 dimensions. The number of dimensions
// of `tensor` must equal the number of elements in `dims`. In other words:
//
// `rank(tensor) = size(dims)`
//
// For example:
//
// ```prettyprint
// # tensor 't' is [[[[ 0,  1,  2,  3],
// #                  [ 4,  5,  6,  7],
// #                  [ 8,  9, 10, 11]],
// #                 [[12, 13, 14, 15],
// #                  [16, 17, 18, 19],
// #                  [20, 21, 22, 23]]]]
// # tensor 't' shape is [1, 2, 3, 4]
//
// # 'dims' is [False, False, False, True]
// reverse(t, dims) ==> [[[[ 3,  2,  1,  0],
//                         [ 7,  6,  5,  4],
//                         [ 11, 10, 9, 8]],
//                        [[15, 14, 13, 12],
//                         [19, 18, 17, 16],
//                         [23, 22, 21, 20]]]]
//
// # 'dims' is [False, True, False, False]
// reverse(t, dims) ==> [[[[12, 13, 14, 15],
//                         [16, 17, 18, 19],
//                         [20, 21, 22, 23]
//                        [[ 0,  1,  2,  3],
//                         [ 4,  5,  6,  7],
//                         [ 8,  9, 10, 11]]]]
//
// # 'dims' is [False, False, True, False]
// reverse(t, dims) ==> [[[[8, 9, 10, 11],
//                         [4, 5, 6, 7],
//                         [0, 1, 2, 3]]
//                        [[20, 21, 22, 23],
//                         [16, 17, 18, 19],
//                         [12, 13, 14, 15]]]]
// ```
//
// Arguments:
// * tensor: Up to 8-D.
// * dims: 1-D. The dimensions to reverse.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The same shape as `tensor`.
Node* Reverse(NodeOut tensor, NodeOut dims, const GraphDefBuilder::Options&
              opts);

// Reverses variable length slices.
//
// This op first slices `input` along the dimension `batch_dim`, and for each
// slice `i`, reverses the first `seq_lengths[i]` elements along
// the dimension `seq_dim`.
//
// The elements of `seq_lengths` must obey `seq_lengths[i] < input.dims[seq_dim]`,
// and `seq_lengths` must be a vector of length `input.dims[batch_dim]`.
//
// The output slice `i` along dimension `batch_dim` is then given by input
// slice `i`, with the first `seq_lengths[i]` slices along dimension
// `seq_dim` reversed.
//
// For example:
//
// ```prettyprint
// # Given this:
// batch_dim = 0
// seq_dim = 1
// input.dims = (4, 8, ...)
// seq_lengths = [7, 2, 3, 5]
//
// # then slices of input are reversed on seq_dim, but only up to seq_lengths:
// output[0, 0:7, :, ...] = input[0, 7:0:-1, :, ...]
// output[1, 0:2, :, ...] = input[1, 2:0:-1, :, ...]
// output[2, 0:3, :, ...] = input[2, 3:0:-1, :, ...]
// output[3, 0:5, :, ...] = input[3, 5:0:-1, :, ...]
//
// # while entries past seq_lens are copied through:
// output[0, 7:, :, ...] = input[0, 7:, :, ...]
// output[1, 2:, :, ...] = input[1, 2:, :, ...]
// output[2, 3:, :, ...] = input[2, 3:, :, ...]
// output[3, 2:, :, ...] = input[3, 2:, :, ...]
// ```
//
// In contrast, if:
//
// ```prettyprint
// # Given this:
// batch_dim = 2
// seq_dim = 0
// input.dims = (8, ?, 4, ...)
// seq_lengths = [7, 2, 3, 5]
//
// # then slices of input are reversed on seq_dim, but only up to seq_lengths:
// output[0:7, :, 0, :, ...] = input[7:0:-1, :, 0, :, ...]
// output[0:2, :, 1, :, ...] = input[2:0:-1, :, 1, :, ...]
// output[0:3, :, 2, :, ...] = input[3:0:-1, :, 2, :, ...]
// output[0:5, :, 3, :, ...] = input[5:0:-1, :, 3, :, ...]
//
// # while entries past seq_lens are copied through:
// output[7:, :, 0, :, ...] = input[7:, :, 0, :, ...]
// output[2:, :, 1, :, ...] = input[2:, :, 1, :, ...]
// output[3:, :, 2, :, ...] = input[3:, :, 2, :, ...]
// output[2:, :, 3, :, ...] = input[2:, :, 3, :, ...]
// ```
//
// Arguments:
// * input: The input to reverse.
// * seq_lengths: 1-D with length `input.dims(batch_dim)` and
// `max(seq_lengths) < input.dims(seq_dim)`
// * seq_dim: The dimension which is partially reversed.
// * opts:
//   .WithAttr("batch_dim", int64): Defaults to 0.
//     The dimension along which reversal is performed.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The partially reversed input. It has the same shape as `input`.
Node* ReverseSequence(NodeOut input, NodeOut seq_lengths, int64 seq_dim, const
                      GraphDefBuilder::Options& opts);

// Returns the shape of a tensor.
//
// This operation returns a 1-D integer tensor representing the shape of `input`.
//
// For example:
//
// ```prettyprint
// # 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
// shape(t) ==> [2, 2, 3]
// ```
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* Shape(NodeOut input, const GraphDefBuilder::Options& opts);

// Returns shape of tensors.
//
// This operation returns N 1-D integer tensors representing shape of `input[i]s`.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* ShapeN(gtl::ArraySlice<NodeOut> input, const GraphDefBuilder::Options&
             opts);

// Returns the size of a tensor.
//
// This operation returns an integer representing the number of elements in
// `input`.
//
// For example:
//
// ```prettyprint
// # 't' is [[[1, 1,, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]]
// size(t) ==> 12
// ```
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* Size(NodeOut input, const GraphDefBuilder::Options& opts);

// Return a slice from 'input'.
//
// The output tensor is a tensor with dimensions described by 'size'
// whose values are extracted from 'input' starting at the offsets in
// 'begin'.
//
// *Requirements*:
//   0 <= begin[i] <= begin[i] + size[i] <= Di  for i in [0, n)
//
// Arguments:
// * begin: begin[i] specifies the offset into the 'i'th dimension of
// 'input' to slice from.
// * size: size[i] specifies the number of elements of the 'i'th dimension
// of 'input' to slice. If size[i] is -1, all remaining elements in dimension
// i are included in the slice (i.e. this is equivalent to setting
// size[i] = input.dim_size(i) - begin[i]).
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* Slice(NodeOut input, NodeOut begin, NodeOut size, const
            GraphDefBuilder::Options& opts);

// SpaceToDepth for tensors of type T.
//
// Rearranges blocks of spatial data, into depth. More specifically,
// this op outputs a copy of the input tensor where values from the `height`
// and `width` dimensions are moved to the `depth` dimension.
// The attr `block_size` indicates the input block size and how the data is moved.
//
//   * Non-overlapping blocks of size `block_size x block size` are rearranged
//     into depth at each location.
//   * The depth of the output tensor is `input_depth * block_size * block_size`.
//   * The input tensor's height and width must be divisible by block_size.
//
// That is, assuming the input is in the shape:
// `[batch, height, width, depth]`,
// the shape of the output will be:
// `[batch, height/block_size, width/block_size, depth*block_size*block_size]`
//
// This operation requires that the input tensor be of rank 4, and that
// `block_size` be >=1 and a divisor of both the input `height` and `width`.
//
// This operation is useful for resizing the activations between convolutions
// (but keeping all data), e.g. instead of pooling. It is also useful for training
// purely convolutional models.
//
// For example, given this input of shape `[1, 2, 2, 1]`, and block_size of 2:
//
// ```prettyprint
// x = [[[[1], [2]],
//       [[3], [4]]]]
// ```
//
// This operation will output a tensor of shape `[1, 1, 1, 4]`:
//
// ```prettyprint
// [[[[1, 2, 3, 4]]]]
// ```
//
// Here, the input has a batch of 1 and each batch element has shape `[2, 2, 1]`,
// the corresponding output will have a single element (i.e. width and height are
// both 1) and will have a depth of 4 channels (1 * block_size * block_size).
// The output element shape is `[1, 1, 4]`.
//
// For an input tensor with larger depth, here of shape `[1, 2, 2, 3]`, e.g.
//
// ```prettyprint
// x = [[[[1, 2, 3], [4, 5, 6]],
//       [[7, 8, 9], [10, 11, 12]]]]
// ```
//
// This operation, for block_size of 2, will return the following tensor of shape
// `[1, 1, 1, 12]`
//
// ```prettyprint
// [[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]]]
// ```
//
// Similarly, for the following input of shape `[1 4 4 1]`, and a block size of 2:
//
// ```prettyprint
// x = [[ [1],   [2],  [5],  [6]],
//      [ [3],   [4],  [7],  [8]],
//      [ [9],  [10], [13],  [14]],
//      [ [11], [12], [15],  [16]]]
// ```
//
// the operator will return the following tensor of shape `[1 2 2 4]`:
//
// ```prettyprint
// x = [[[[1, 2, 3, 4],
//        [5, 6, 7, 8]],
//       [[9, 10, 11, 12],
//        [13, 14, 15, 16]]]]
// ```
//
// Arguments:
// * block_size: The size of the spatial block.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* SpaceToDepth(NodeOut input, int64 block_size, const
                   GraphDefBuilder::Options& opts);

// Splits a tensor into `num_split` tensors along one dimension.
//
// Arguments:
// * split_dim: 0-D.  The dimension along which to split.  Must be in the range
// `[0, rank(value))`.
// * value: The tensor to split.
// * num_split: The number of ways to split.  Must evenly divide
// `value.shape[split_dim]`.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// They are identically shaped tensors, whose shape matches that of `value`
// except along `split_dim`, where their sizes are
// `values.shape[split_dim] / num_split`.
Node* Split(NodeOut split_dim, NodeOut value, int64 num_split, const
            GraphDefBuilder::Options& opts);

// Removes dimensions of size 1 from the shape of a tensor.
//
// Given a tensor `input`, this operation returns a tensor of the same type with
// all dimensions of size 1 removed. If you don't want to remove all size 1
// dimensions, you can remove specific size 1 dimensions by specifying
// `squeeze_dims`.
//
// For example:
//
// ```prettyprint
// # 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
// shape(squeeze(t)) ==> [2, 3]
// ```
//
// Or, to remove specific size 1 dimensions:
//
// ```prettyprint
// # 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
// shape(squeeze(t, [2, 4])) ==> [1, 2, 3, 1]
// ```
//
// Arguments:
// * input: The `input` to squeeze.
// * opts:
//   .WithAttr("squeeze_dims", gtl::ArraySlice<int>): Defaults to [].
//     If specified, only squeezes the dimensions listed. The dimension
// index starts at 0. It is an error to squeeze a dimension that is not 1.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Contains the same data as `input`, but has one or more dimensions of
// size 1 removed.
Node* Squeeze(NodeOut input, const GraphDefBuilder::Options& opts);

// Stops gradient computation.
//
// When executed in a graph, this op outputs its input tensor as-is.
//
// When building ops to compute gradients, this op prevents the contribution of
// its inputs to be taken into account.  Normally, the gradient generator adds ops
// to a graph to compute the derivatives of a specified 'loss' by recursively
// finding out inputs that contributed to its computation.  If you insert this op
// in the graph it inputs are masked from the gradient generator.  They are not
// taken into account for computing gradients.
//
// This is useful any time you want to compute a value with TensorFlow but need
// to pretend that the value was a constant. Some examples include:
//
// *  The *EM* algorithm where the *M-step* should not involve backpropagation
//    through the output of the *E-step*.
// *  Contrastive divergence training of Boltzmann machines where, when
//    differentiating the energy function, the training must not backpropagate
//    through the graph that generated the samples from the model.
// *  Adversarial training, where no backprop should happen through the adversarial
//    example generation process.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* StopGradient(NodeOut input, const GraphDefBuilder::Options& opts);

// Constructs a tensor by tiling a given tensor.
//
// This operation creates a new tensor by replicating `input` `multiples` times.
// The output tensor's i'th dimension has `input.dims(i) * multiples[i]` elements,
// and the values of `input` are replicated `multiples[i]` times along the 'i'th
// dimension. For example, tiling `[a b c d]` by `[2]` produces
// `[a b c d a b c d]`.
//
// Arguments:
// * input: 1-D or higher.
// * multiples: 1-D. Length must be the same as the number of dimensions in `input`
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* Tile(NodeOut input, NodeOut multiples, const GraphDefBuilder::Options&
           opts);

// Returns the gradient of `Tile`.
//
// Since `Tile` takes an input and repeats the input `multiples` times
// along each dimension, `TileGrad` takes in `multiples` and aggregates
// each repeated tile of `input` into `output`.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* TileGrad(NodeOut input, NodeOut multiples, const
               GraphDefBuilder::Options& opts);

// Shuffle dimensions of x according to a permutation.
//
// The output `y` has the same rank as `x`. The shapes of `x` and `y` satisfy:
//   `y.shape[i] == x.shape[perm[i]] for i in [0, 1, ..., rank(x) - 1]`
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* Transpose(NodeOut x, NodeOut perm, const GraphDefBuilder::Options& opts);

// Finds unique elements in a 1-D tensor.
//
// This operation returns a tensor `y` containing all of the unique elements of `x`
// sorted in the same order that they occur in `x`. This operation also returns a
// tensor `idx` the same size as `x` that contains the index of each value of `x`
// in the unique output `y`. In other words:
//
// `y[idx[i]] = x[i] for i in [0, 1,...,rank(x) - 1]`
//
// For example:
//
// ```prettyprint
// # tensor 'x' is [1, 1, 2, 4, 4, 4, 7, 8, 8]
// y, idx = unique(x)
// y ==> [1, 2, 4, 7, 8]
// idx ==> [0, 0, 1, 2, 2, 2, 3, 4, 4]
// ```
//
// Arguments:
// * x: 1-D.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with outputs:
// * y: 1-D.
// * idx: 1-D.
Node* Unique(NodeOut x, const GraphDefBuilder::Options& opts);

// Finds unique elements in a 1-D tensor.
//
// This operation returns a tensor `y` containing all of the unique elements of `x`
// sorted in the same order that they occur in `x`. This operation also returns a
// tensor `idx` the same size as `x` that contains the index of each value of `x`
// in the unique output `y`. Finally, it returns a third tensor `count` that
// contains the count of each element of `y` in `x`. In other words:
//
// `y[idx[i]] = x[i] for i in [0, 1,...,rank(x) - 1]`
//
// For example:
//
// ```prettyprint
// # tensor 'x' is [1, 1, 2, 4, 4, 4, 7, 8, 8]
// y, idx, count = unique(x)
// y ==> [1, 2, 4, 7, 8]
// idx ==> [0, 0, 1, 2, 2, 2, 3, 4, 4]
// count ==> [2, 1, 3, 1, 2]
// ```
//
// Arguments:
// * x: 1-D.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with outputs:
// * y: 1-D.
// * idx: 1-D.
// * count: 1-D.
Node* UniqueWithCounts(NodeOut x, const GraphDefBuilder::Options& opts);

// Unpacks the outer dimension of a rank-`R` tensor into `num` rank-`(R-1)` tensors.
//
// Unpacks `num` tensors from `value` by chipping it along the first dimension.
// The i'th tensor in `output` is the slice `value[i, ...]`. Each tensor in
// `output` has shape `value.shape[1:]`.
//
// This is the opposite of `pack`.
//
// Arguments:
// * value: 1-D or higher, with first dimension `num`.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The list of tensors unpacked from `value`.
Node* Unpack(NodeOut value, int64 num, const GraphDefBuilder::Options& opts);

// Returns locations of true values in a boolean tensor.
//
// This operation returns the coordinates of true elements in `input`. The
// coordinates are returned in a 2-D tensor where the first dimension (rows)
// represents the number of true elements, and the second dimension (columns)
// represents the coordinates of the true elements. Keep in mind, the shape of
// the output tensor can vary depending on how many true values there are in
// `input`. Indices are output in row-major order.
//
// For example:
//
// ```prettyprint
// # 'input' tensor is [[True, False]
// #                    [True, False]]
// # 'input' has two true values, so output has two coordinates.
// # 'input' has rank of 2, so coordinates have two indices.
// where(input) ==> [[0, 0],
//                   [1, 0]]
//
// # `input` tensor is [[[True, False]
// #                     [True, False]]
// #                    [[False, True]
// #                     [False, True]]
// #                    [[False, False]
// #                     [False, True]]]
// # 'input' has 5 true values, so output has 5 coordinates.
// # 'input' has rank of 3, so coordinates have three indices.
// where(input) ==> [[0, 0, 0],
//                   [0, 1, 0],
//                   [1, 0, 1],
//                   [1, 1, 1],
//                   [2, 1, 1]]
// ```
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* Where(NodeOut input, const GraphDefBuilder::Options& opts);

// Returns a tensor of zeros with the same shape and type as x.
//
// Arguments:
// * x: a tensor of type T.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// a tensor of the same shape and type as x but filled with zeros.
Node* ZerosLike(NodeOut x, const GraphDefBuilder::Options& opts);

}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_OPS_ARRAY_OPS_H_
