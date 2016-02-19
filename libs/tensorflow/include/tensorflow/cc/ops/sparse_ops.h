// This file is MACHINE GENERATED! Do not edit.

#ifndef TENSORFLOW_CC_OPS_SPARSE_OPS_H_
#define TENSORFLOW_CC_OPS_SPARSE_OPS_H_

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


// Deserialize and concatenate `SparseTensors` from a serialized minibatch.
//
// The input `serialized_sparse` must be a string matrix of shape `[N x 3]` where
// `N` is the minibatch size and the rows correspond to packed outputs of
// `SerializeSparse`.  The ranks of the original `SparseTensor` objects
// must all match.  When the final `SparseTensor` is created, it has rank one
// higher than the ranks of the incoming `SparseTensor` objects
// (they have been concatenated along a new row dimension).
//
// The output `SparseTensor` object's shape values for all dimensions but the
// first are the max across the input `SparseTensor` objects' shape values
// for the corresponding dimensions.  Its first shape value is `N`, the minibatch
// size.
//
// The input `SparseTensor` objects' indices are assumed ordered in
// standard lexicographic order.  If this is not the case, after this
// step run `SparseReorder` to restore index ordering.
//
// For example, if the serialized input is a `[2 x 3]` matrix representing two
// original `SparseTensor` objects:
//
//     index = [ 0]
//             [10]
//             [20]
//     values = [1, 2, 3]
//     shape = [50]
//
// and
//
//     index = [ 2]
//             [10]
//     values = [4, 5]
//     shape = [30]
//
// then the final deserialized `SparseTensor` will be:
//
//     index = [0  0]
//             [0 10]
//             [0 20]
//             [1  2]
//             [1 10]
//     values = [1, 2, 3, 4, 5]
//     shape = [2 50]
//
// Arguments:
// * serialized_sparse: 2-D, The `N` serialized `SparseTensor` objects.
// Must have 3 columns.
// * dtype: The `dtype` of the serialized `SparseTensor` objects.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with outputs:
// * sparse_indices
// * sparse_values
// * sparse_shape
Node* DeserializeManySparse(NodeOut serialized_sparse, DataType dtype, const
                            GraphDefBuilder::Options& opts);

// Serialize an `N`-minibatch `SparseTensor` into an `[N, 3]` string `Tensor`.
//
// The `SparseTensor` must have rank `R` greater than 1, and the first dimension
// is treated as the minibatch dimension.  Elements of the `SparseTensor`
// must be sorted in increasing order of this first dimension.  The serialized
// `SparseTensor` objects going into each row of `serialized_sparse` will have
// rank `R-1`.
//
// The minibatch size `N` is extracted from `sparse_shape[0]`.
//
// Arguments:
// * sparse_indices: 2-D.  The `indices` of the minibatch `SparseTensor`.
// * sparse_values: 1-D.  The `values` of the minibatch `SparseTensor`.
// * sparse_shape: 1-D.  The `shape` of the minibatch `SparseTensor`.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* SerializeManySparse(NodeOut sparse_indices, NodeOut sparse_values,
                          NodeOut sparse_shape, const GraphDefBuilder::Options&
                          opts);

// Serialize a `SparseTensor` into a string 3-vector (1-D `Tensor`) object.
//
// Arguments:
// * sparse_indices: 2-D.  The `indices` of the `SparseTensor`.
// * sparse_values: 1-D.  The `values` of the `SparseTensor`.
// * sparse_shape: 1-D.  The `shape` of the `SparseTensor`.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* SerializeSparse(NodeOut sparse_indices, NodeOut sparse_values, NodeOut
                      sparse_shape, const GraphDefBuilder::Options& opts);

// Concatenates a list of `SparseTensor` along the specified dimension.
//
// Concatenation is with respect to the dense versions of these sparse tensors.
// It is assumed that each input is a `SparseTensor` whose elements are ordered
// along increasing dimension number.
//
// All inputs' shapes must match, except for the concat dimension.  The
// `indices`, `values`, and `shapes` lists must have the same length.
//
// The output shape is identical to the inputs', except along the concat
// dimension, where it is the sum of the inputs' sizes along that dimension.
//
// The output elements will be resorted to preserve the sort order along
// increasing dimension number.
//
// This op runs in `O(M log M)` time, where `M` is the total number of non-empty
// values across all inputs. This is due to the need for an internal sort in
// order to concatenate efficiently across an arbitrary dimension.
//
// For example, if `concat_dim = 1` and the inputs are
//
//     sp_inputs[0]: shape = [2, 3]
//     [0, 2]: "a"
//     [1, 0]: "b"
//     [1, 1]: "c"
//
//     sp_inputs[1]: shape = [2, 4]
//     [0, 1]: "d"
//     [0, 2]: "e"
//
// then the output will be
//
//     shape = [2, 7]
//     [0, 2]: "a"
//     [0, 4]: "d"
//     [0, 5]: "e"
//     [1, 0]: "b"
//     [1, 1]: "c"
//
// Graphically this is equivalent to doing
//
//     [    a] concat [  d e  ] = [    a   d e  ]
//     [b c  ]        [       ]   [b c          ]
//
// Arguments:
// * indices: 2-D.  Indices of each input `SparseTensor`.
// * values: 1-D.  Non-empty values of each `SparseTensor`.
// * shapes: 1-D.  Shapes of each `SparseTensor`.
// * concat_dim: Dimension to concatenate along.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with outputs:
// * output_indices: 2-D.  Indices of the concatenated `SparseTensor`.
// * output_values: 1-D.  Non-empty values of the concatenated `SparseTensor`.
// * output_shape: 1-D.  Shape of the concatenated `SparseTensor`.
Node* SparseConcat(gtl::ArraySlice<NodeOut> indices, gtl::ArraySlice<NodeOut>
                   values, gtl::ArraySlice<NodeOut> shapes, int64 concat_dim,
                   const GraphDefBuilder::Options& opts);

// Reorders a SparseTensor into the canonical, row-major ordering.
//
// Note that by convention, all sparse ops preserve the canonical ordering along
// increasing dimension number. The only time ordering can be violated is during
// manual manipulation of the indices and values vectors to add entries.
//
// Reordering does not affect the shape of the SparseTensor.
//
// If the tensor has rank `R` and `N` non-empty values, `input_indices` has
// shape `[N, R]`, input_values has length `N`, and input_shape has length `R`.
//
// Arguments:
// * input_indices: 2-D.  `N x R` matrix with the indices of non-empty values in a
// SparseTensor, possibly not in canonical ordering.
// * input_values: 1-D.  `N` non-empty values corresponding to `input_indices`.
// * input_shape: 1-D.  Shape of the input SparseTensor.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with outputs:
// * output_indices: 2-D.  `N x R` matrix with the same indices as input_indices, but
// in canonical row-major ordering.
// * output_values: 1-D.  `N` non-empty values corresponding to `output_indices`.
Node* SparseReorder(NodeOut input_indices, NodeOut input_values, NodeOut
                    input_shape, const GraphDefBuilder::Options& opts);

// Split a `SparseTensor` into `num_split` tensors along one dimension.
//
// If the `shape[split_dim]` is not an integer multiple of `num_split`. Slices
// `[0 : shape[split_dim] % num_split]` gets one extra dimension.
// For example, if `split_dim = 1` and `num_split = 2` and the input is
//
//     input_tensor = shape = [2, 7]
//     [    a   d e  ]
//     [b c          ]
//
// Graphically the output tensors are:
//
//     output_tensor[0] = shape = [2, 4]
//     [    a  ]
//     [b c    ]
//
//     output_tensor[1] = shape = [2, 3]
//     [ d e  ]
//     [      ]
//
// Arguments:
// * split_dim: 0-D.  The dimension along which to split.  Must be in the range
// `[0, rank(shape))`.
// * indices: 2-D tensor represents the indices of the sparse tensor.
// * values: 1-D tensor represents the values of the sparse tensor.
// * shape: 1-D. tensor represents the shape of the sparse tensor.
// output indices: A list of 1-D tensors represents the indices of the output
// sparse tensors.
// * num_split: The number of ways to split.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with outputs:
// * output_indices
// * output_values: A list of 1-D tensors represents the values of the output sparse
// tensors.
// * output_shape: A list of 1-D tensors represents the shape of the output sparse
// tensors.
Node* SparseSplit(NodeOut split_dim, NodeOut indices, NodeOut values, NodeOut
                  shape, int64 num_split, const GraphDefBuilder::Options&
                  opts);

// Multiply SparseTensor (of rank 2) "A" by dense matrix "B".
//
// No validity checking is performed on the indices of A.  However, the following
// input format is recommended for optimal behavior:
//
// if adjoint_a == false:
//   A should be sorted in lexicographically increasing order.  Use SparseReorder
//   if you're not sure.
// if adjoint_a == true:
//   A should be sorted in order of increasing dimension 1 (i.e., "column major"
//   order instead of "row major" order).
//
// Arguments:
// * a_indices: 2-D.  The `indices` of the `SparseTensor`, size [nnz x 2] Matrix.
// * a_values: 1-D.  The `values` of the `SparseTensor`, size [nnz] Vector.
// * a_shape: 1-D.  The `shape` of the `SparseTensor`, size [2] Vector.
// * b: 2-D.  A dense Matrix.
// * opts:
//   .WithAttr("adjoint_a", bool): Defaults to false.
//     Use the adjoint of A in the matrix multiply.  If A is complex, this
// is transpose(conj(A)).  Otherwise it's transpose(A).
//   .WithAttr("adjoint_b", bool): Defaults to false.
//     Use the adjoint of B in the matrix multiply.  If B is complex, this
// is transpose(conj(B)).  Otherwise it's transpose(B).
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* SparseTensorDenseMatMul(NodeOut a_indices, NodeOut a_values, NodeOut
                              a_shape, NodeOut b, const
                              GraphDefBuilder::Options& opts);

// Converts a sparse representation into a dense tensor.
//
// Builds an array `dense` with shape `output_shape` such that
//
// ```prettyprint
// # If sparse_indices is scalar
// dense[i] = (i == sparse_indices ? sparse_values : default_value)
//
// # If sparse_indices is a vector, then for each i
// dense[sparse_indices[i]] = sparse_values[i]
//
// # If sparse_indices is an n by d matrix, then for each i in [0, n)
// dense[sparse_indices[i][0], ..., sparse_indices[i][d-1]] = sparse_values[i]
// ```
//
// All other values in `dense` are set to `default_value`.  If `sparse_values` is a
// scalar, all sparse indices are set to this single value.
//
// Indices should be sorted in lexicographic order, and indices must not
// contain any repeats. If `validate_indices` is true, these properties
// are checked during execution.
//
// Arguments:
// * sparse_indices: 0-D, 1-D, or 2-D.  `sparse_indices[i]` contains the complete
// index where `sparse_values[i]` will be placed.
// * output_shape: 1-D.  Shape of the dense output tensor.
// * sparse_values: 1-D.  Values corresponding to each row of `sparse_indices`,
// or a scalar value to be used for all sparse indices.
// * default_value: Scalar value to set for indices not specified in
// `sparse_indices`.
// * opts:
//   .WithAttr("validate_indices", bool): Defaults to true.
//     If true, indices are checked to make sure they are sorted in
// lexicographic order and that there are no repeats.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Dense output tensor of shape `output_shape`.
Node* SparseToDense(NodeOut sparse_indices, NodeOut output_shape, NodeOut
                    sparse_values, NodeOut default_value, const
                    GraphDefBuilder::Options& opts);

}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_OPS_SPARSE_OPS_H_
