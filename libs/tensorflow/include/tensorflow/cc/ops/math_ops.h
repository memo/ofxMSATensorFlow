// This file is MACHINE GENERATED! Do not edit.

#ifndef TENSORFLOW_CC_OPS_MATH_OPS_H_
#define TENSORFLOW_CC_OPS_MATH_OPS_H_

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


// Computes the absolute value of a tensor.
//
// Given a tensor `x`, this operation returns a tensor containing the absolute
// value of each element in `x`. For example, if x is an input element and y is
// an output element, this operation computes \\(y = |x|\\).
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* Abs(NodeOut x, const GraphDefBuilder::Options& opts);

// Returns x + y element-wise.
//
// *NOTE*: Add supports broadcasting. AddN does not.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* Add(NodeOut x, NodeOut y, const GraphDefBuilder::Options& opts);

// Add all input tensors element wise.
//
// Arguments:
// * inputs: Must all be the same size and shape.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* AddN(gtl::ArraySlice<NodeOut> inputs, const GraphDefBuilder::Options&
           opts);

// Computes the "logical and" of elements across dimensions of a tensor.
//
// Reduces `input` along the dimensions given in `reduction_indices`. Unless
// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
// `reduction_indices`. If `keep_dims` is true, the reduced dimensions are
// retained with length 1.
//
// Arguments:
// * input: The tensor to reduce.
// * reduction_indices: The dimensions to reduce.
// * opts:
//   .WithAttr("keep_dims", bool): Defaults to false.
//     If true, retain reduced dimensions with length 1.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The reduced tensor.
Node* All(NodeOut input, NodeOut reduction_indices, const
          GraphDefBuilder::Options& opts);

// Computes the "logical or" of elements across dimensions of a tensor.
//
// Reduces `input` along the dimensions given in `reduction_indices`. Unless
// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
// `reduction_indices`. If `keep_dims` is true, the reduced dimensions are
// retained with length 1.
//
// Arguments:
// * input: The tensor to reduce.
// * reduction_indices: The dimensions to reduce.
// * opts:
//   .WithAttr("keep_dims", bool): Defaults to false.
//     If true, retain reduced dimensions with length 1.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The reduced tensor.
Node* Any(NodeOut input, NodeOut reduction_indices, const
          GraphDefBuilder::Options& opts);

// Returns the index with the largest value across dimensions of a tensor.
//
// Arguments:
// * dimension: int32, 0 <= dimension < rank(input).  Describes which dimension
// of the input Tensor to reduce across. For vectors, use dimension = 0.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* ArgMax(NodeOut input, NodeOut dimension, const GraphDefBuilder::Options&
             opts);

// Returns the index with the smallest value across dimensions of a tensor.
//
// Arguments:
// * dimension: int32, 0 <= dimension < rank(input).  Describes which dimension
// of the input Tensor to reduce across. For vectors, use dimension = 0.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* ArgMin(NodeOut input, NodeOut dimension, const GraphDefBuilder::Options&
             opts);

// Multiplies slices of two tensors in batches.
//
// Multiplies all slices of `Tensor` `x` and `y` (each slice can be
// viewed as an element of a batch), and arranges the individual results
// in a single output tensor of the same batch size. Each of the
// individual slices can optionally be adjointed (to adjoint a matrix
// means to transpose and conjugate it) before multiplication by setting
// the `adj_x` or `adj_y` flag to `True`, which are by default `False`.
//
// The input tensors `x` and `y` are 3-D or higher with shape `[..., r_x, c_x]`
// and `[..., r_y, c_y]`.
//
// The output tensor is 3-D or higher with shape `[..., r_o, c_o]`, where:
//
//     r_o = c_x if adj_x else r_x
//     c_o = r_y if adj_y else c_y
//
// It is computed as:
//
//     out[..., :, :] = matrix(x[..., :, :]) * matrix(y[..., :, :])
//
// Arguments:
// * x: 3-D or higher with shape `[..., r_x, c_x]`.
// * y: 3-D or higher with shape `[..., r_y, c_y]`.
// * opts:
//   .WithAttr("adj_x", bool): Defaults to false.
//     If `True`, adjoint the slices of `x`. Defaults to `False`.
//   .WithAttr("adj_y", bool): Defaults to false.
//     If `True`, adjoint the slices of `y`. Defaults to `False`.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// 3-D or higher with shape `[..., r_o, c_o]`
Node* BatchMatMul(NodeOut x, NodeOut y, const GraphDefBuilder::Options& opts);

// Cast x of type SrcT to y of DstT.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* Cast(NodeOut x, DataType DstT, const GraphDefBuilder::Options& opts);

// Returns element-wise smallest integer in not less than x.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* Ceil(NodeOut x, const GraphDefBuilder::Options& opts);

// Converts two real numbers to a complex number.
//
// Given a tensor `real` representing the real part of a complex number, and a
// tensor `imag` representing the imaginary part of a complex number, this
// operation returns complex numbers elementwise of the form \\(a + bj\\), where
// *a* represents the `real` part and *b* represents the `imag` part.
//
// The input tensors `real` and `imag` must have the same shape.
//
// For example:
//
// ```
// # tensor 'real' is [2.25, 3.25]
// # tensor `imag` is [4.75, 5.75]
// tf.complex(real, imag) ==> [[2.25 + 4.75j], [3.25 + 5.75j]]
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
Node* Complex(NodeOut real, NodeOut imag, const GraphDefBuilder::Options&
              opts);

// Computes the complex absolute value of a tensor.
//
// Given a tensor `x` of complex numbers, this operation returns a tensor of type
// `float` that is the absolute value of each element in `x`. All elements in `x`
// must be complex numbers of the form \\(a + bj\\). The absolute value is
// computed as \\( \sqrt{a^2 + b^2}\\).
//
// For example:
//
// ```
// # tensor 'x' is [[-2.25 + 4.75j], [-3.25 + 5.75j]]
// tf.complex_abs(x) ==> [5.25594902, 6.60492229]
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
Node* ComplexAbs(NodeOut x, const GraphDefBuilder::Options& opts);

// Returns the complex conjugate of a complex number.
//
// Given a tensor `in` of complex numbers, this operation returns a tensor of
// complex numbers that are the complex conjugate of each element in `in`. The
// complex numbers in `in` must be of the form \\(a + bj\\), where *a* is the real
// part and *b* is the imaginary part.
//
// The complex conjugate returned by this operation is of the form \\(a - bj\\).
//
// For example:
//
// ```
// # tensor 'in' is [-2.25 + 4.75j, 3.25 + 5.75j]
// tf.conj(in) ==> [-2.25 - 4.75j, 3.25 - 5.75j]
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
Node* Conj(NodeOut in, const GraphDefBuilder::Options& opts);

// Computes cos of x element-wise.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* Cos(NodeOut x, const GraphDefBuilder::Options& opts);

// Returns x / y element-wise.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* Div(NodeOut x, NodeOut y, const GraphDefBuilder::Options& opts);

// Returns the truth value of (x == y) element-wise.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* Equal(NodeOut x, NodeOut y, const GraphDefBuilder::Options& opts);

// Computes the Gauss error function of `x` element-wise.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* Erf(NodeOut x, const GraphDefBuilder::Options& opts);

// Computes the complementary error function of `x` element-wise.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* Erfc(NodeOut x, const GraphDefBuilder::Options& opts);

// Computes exponential of x element-wise.  \\(y = e^x\\).
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* Exp(NodeOut x, const GraphDefBuilder::Options& opts);

// Compute the 2-dimensional discrete Fourier Transform.
//
// Arguments:
// * in: A complex64 matrix.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The 2D Fourier Transform of `in`.
Node* FFT2D(NodeOut in, const GraphDefBuilder::Options& opts);

// Returns element-wise largest integer not greater than x.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* Floor(NodeOut x, const GraphDefBuilder::Options& opts);

// Returns the truth value of (x > y) element-wise.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* Greater(NodeOut x, NodeOut y, const GraphDefBuilder::Options& opts);

// Returns the truth value of (x >= y) element-wise.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* GreaterEqual(NodeOut x, NodeOut y, const GraphDefBuilder::Options& opts);

// Compute the inverse 2-dimensional discrete Fourier Transform.
//
// Arguments:
// * in: A complex64 matrix.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The inverse 2D Fourier Transform of `in`.
Node* IFFT2D(NodeOut in, const GraphDefBuilder::Options& opts);

// Returns the imaginary part of a complex number.
//
// Given a tensor `in` of complex numbers, this operation returns a tensor of type
// `float` that is the imaginary part of each element in `in`. All elements in `in`
// must be complex numbers of the form \\(a + bj\\), where *a* is the real part
// and *b* is the imaginary part returned by this operation.
//
// For example:
//
// ```
// # tensor 'in' is [-2.25 + 4.75j, 3.25 + 5.75j]
// tf.imag(in) ==> [4.75, 5.75]
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
Node* Imag(NodeOut in, const GraphDefBuilder::Options& opts);

// Computes the reciprocal of x element-wise.
//
// I.e., \\(y = 1 / x\\).
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* Inv(NodeOut x, const GraphDefBuilder::Options& opts);

// Returns which elements of x are finite.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* IsFinite(NodeOut x, const GraphDefBuilder::Options& opts);

// Returns which elements of x are Inf.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* IsInf(NodeOut x, const GraphDefBuilder::Options& opts);

// Returns which elements of x are NaN.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* IsNan(NodeOut x, const GraphDefBuilder::Options& opts);

// Returns the truth value of (x < y) element-wise.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* Less(NodeOut x, NodeOut y, const GraphDefBuilder::Options& opts);

// Returns the truth value of (x <= y) element-wise.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* LessEqual(NodeOut x, NodeOut y, const GraphDefBuilder::Options& opts);

// Computes the log of the absolute value of Gamma of `x` element-wise.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* Lgamma(NodeOut x, const GraphDefBuilder::Options& opts);

// Generates values in an interval.
//
// A sequence of `num` evenly-spaced values are generated beginning at `start`.
// If `num > 1`, the values in the sequence increase by `stop - start / num - 1`,
// so that the last one is exactly `stop`.
//
// For example:
//
// ```
// tf.linspace(10.0, 12.0, 3, name="linspace") => [ 10.0  11.0  12.0]
// ```
//
// Arguments:
// * start: First entry in the range.
// * stop: Last entry in the range.
// * num: Number of values to generate.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// 1-D. The generated values.
Node* LinSpace(NodeOut start, NodeOut stop, NodeOut num, const
               GraphDefBuilder::Options& opts);

// Computes natural logarithm of x element-wise.
//
// I.e., \\(y = \log_e x\\).
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* Log(NodeOut x, const GraphDefBuilder::Options& opts);

// Returns the truth value of x AND y element-wise.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* LogicalAnd(NodeOut x, NodeOut y, const GraphDefBuilder::Options& opts);

// Returns the truth value of NOT x element-wise.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* LogicalNot(NodeOut x, const GraphDefBuilder::Options& opts);

// Returns the truth value of x OR y element-wise.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* LogicalOr(NodeOut x, NodeOut y, const GraphDefBuilder::Options& opts);

// Multiply the matrix "a" by the matrix "b".
//
// The inputs must be two-dimensional matrices and the inner dimension of
// "a" (after being transposed if transpose_a is true) must match the
// outer dimension of "b" (after being transposed if transposed_b is
// true).
//
// *Note*: The default kernel implementation for MatMul on GPUs uses
// cublas.
//
// Arguments:
// * opts:
//   .WithAttr("transpose_a", bool): Defaults to false.
//     If true, "a" is transposed before multiplication.
//   .WithAttr("transpose_b", bool): Defaults to false.
//     If true, "b" is transposed before multiplication.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* MatMul(NodeOut a, NodeOut b, const GraphDefBuilder::Options& opts);

// Computes the maximum of elements across dimensions of a tensor.
//
// Reduces `input` along the dimensions given in `reduction_indices`. Unless
// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
// `reduction_indices`. If `keep_dims` is true, the reduced dimensions are
// retained with length 1.
//
// Arguments:
// * input: The tensor to reduce.
// * reduction_indices: The dimensions to reduce.
// * opts:
//   .WithAttr("keep_dims", bool): Defaults to false.
//     If true, retain reduced dimensions with length 1.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The reduced tensor.
Node* Max(NodeOut input, NodeOut reduction_indices, const
          GraphDefBuilder::Options& opts);

// Returns the max of x and y (i.e. x > y ? x : y) element-wise, broadcasts.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* Maximum(NodeOut x, NodeOut y, const GraphDefBuilder::Options& opts);

// Computes the mean of elements across dimensions of a tensor.
//
// Reduces `input` along the dimensions given in `reduction_indices`. Unless
// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
// `reduction_indices`. If `keep_dims` is true, the reduced dimensions are
// retained with length 1.
//
// Arguments:
// * input: The tensor to reduce.
// * reduction_indices: The dimensions to reduce.
// * opts:
//   .WithAttr("keep_dims", bool): Defaults to false.
//     If true, retain reduced dimensions with length 1.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The reduced tensor.
Node* Mean(NodeOut input, NodeOut reduction_indices, const
           GraphDefBuilder::Options& opts);

// Computes the minimum of elements across dimensions of a tensor.
//
// Reduces `input` along the dimensions given in `reduction_indices`. Unless
// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
// `reduction_indices`. If `keep_dims` is true, the reduced dimensions are
// retained with length 1.
//
// Arguments:
// * input: The tensor to reduce.
// * reduction_indices: The dimensions to reduce.
// * opts:
//   .WithAttr("keep_dims", bool): Defaults to false.
//     If true, retain reduced dimensions with length 1.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The reduced tensor.
Node* Min(NodeOut input, NodeOut reduction_indices, const
          GraphDefBuilder::Options& opts);

// Returns the min of x and y (i.e. x < y ? x : y) element-wise, broadcasts.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* Minimum(NodeOut x, NodeOut y, const GraphDefBuilder::Options& opts);

// Returns element-wise remainder of division.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* Mod(NodeOut x, NodeOut y, const GraphDefBuilder::Options& opts);

// Returns x * y element-wise.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* Mul(NodeOut x, NodeOut y, const GraphDefBuilder::Options& opts);

// Computes numerical negative value element-wise.
//
// I.e., \\(y = -x\\).
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* Neg(NodeOut x, const GraphDefBuilder::Options& opts);

// Returns the truth value of (x != y) element-wise.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* NotEqual(NodeOut x, NodeOut y, const GraphDefBuilder::Options& opts);

// Computes the power of one value to another.
//
// Given a tensor `x` and a tensor `y`, this operation computes \\(x^y\\) for
// corresponding elements in `x` and `y`. For example:
//
// ```
// # tensor 'x' is [[2, 2]], [3, 3]]
// # tensor 'y' is [[8, 16], [2, 3]]
// tf.pow(x, y) ==> [[256, 65536], [9, 27]]
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
Node* Pow(NodeOut x, NodeOut y, const GraphDefBuilder::Options& opts);

// Computes the product of elements across dimensions of a tensor.
//
// Reduces `input` along the dimensions given in `reduction_indices`. Unless
// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
// `reduction_indices`. If `keep_dims` is true, the reduced dimensions are
// retained with length 1.
//
// Arguments:
// * input: The tensor to reduce.
// * reduction_indices: The dimensions to reduce.
// * opts:
//   .WithAttr("keep_dims", bool): Defaults to false.
//     If true, retain reduced dimensions with length 1.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The reduced tensor.
Node* Prod(NodeOut input, NodeOut reduction_indices, const
           GraphDefBuilder::Options& opts);

// Creates a sequence of integers.
//
// This operation creates a sequence of integers that begins at `start` and
// extends by increments of `delta` up to but not including `limit`.
//
// For example:
//
// ```
// # 'start' is 3
// # 'limit' is 18
// # 'delta' is 3
// tf.range(start, limit, delta) ==> [3, 6, 9, 12, 15]
// ```
//
// Arguments:
// * start: 0-D (scalar). First entry in the sequence.
// * limit: 0-D (scalar). Upper limit of sequence, exclusive.
// * delta: 0-D (scalar). Optional. Default is 1. Number that increments `start`.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// 1-D.
Node* Range(NodeOut start, NodeOut limit, NodeOut delta, const
            GraphDefBuilder::Options& opts);

// Returns the real part of a complex number.
//
// Given a tensor `in` of complex numbers, this operation returns a tensor of type
// `float` that is the real part of each element in `in`. All elements in `in`
// must be complex numbers of the form \\(a + bj\\), where *a* is the real part
// returned by this operation and *b* is the imaginary part.
//
// For example:
//
// ```
// # tensor 'in' is [-2.25 + 4.75j, 3.25 + 5.75j]
// tf.real(in) ==> [-2.25, 3.25]
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
Node* Real(NodeOut in, const GraphDefBuilder::Options& opts);

// Computes reciprocal of square root of x element-wise.
//
// I.e., \\(y = 1 / \sqrt{x}\\).
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* Rsqrt(NodeOut x, const GraphDefBuilder::Options& opts);

// Computes the maximum along segments of a tensor.
//
// Read [the section on Segmentation](../../api_docs/python/math_ops.md#segmentation)
// for an explanation of segments.
//
// Computes a tensor such that
// \\(output_i = \max_j(data_j)\\) where `max` is over `j` such
// that `segment_ids[j] == i`.
//
// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
// <img style="width:100%" src="../../images/SegmentMax.png" alt>
// </div>
//
// Arguments:
// * segment_ids: A 1-D tensor whose rank is equal to the rank of `data`'s
// first dimension.  Values should be sorted and can be repeated.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Has same shape as data, except for dimension 0 which
// has size `k`, the number of segments.
Node* SegmentMax(NodeOut data, NodeOut segment_ids, const
                 GraphDefBuilder::Options& opts);

// Computes the mean along segments of a tensor.
//
// Read [the section on
// Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
// of segments.
//
// Computes a tensor such that
// \\(output_i = \frac{\sum_j data_j}{N}\\) where `mean` is
// over `j` such that `segment_ids[j] == i` and `N` is the total number of
// values summed.
//
// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
// <img style="width:100%" src="../../images/SegmentMean.png" alt>
// </div>
//
// Arguments:
// * segment_ids: A 1-D tensor whose rank is equal to the rank of `data`'s
// first dimension.  Values should be sorted and can be repeated.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Has same shape as data, except for dimension 0 which
// has size `k`, the number of segments.
Node* SegmentMean(NodeOut data, NodeOut segment_ids, const
                  GraphDefBuilder::Options& opts);

// Computes the minimum along segments of a tensor.
//
// Read [the section on
// Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
// of segments.
//
// Computes a tensor such that
// \\(output_i = \min_j(data_j)\\) where `min` is over `j` such
// that `segment_ids[j] == i`.
//
// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
// <img style="width:100%" src="../../images/SegmentMin.png" alt>
// </div>
//
// Arguments:
// * segment_ids: A 1-D tensor whose rank is equal to the rank of `data`'s
// first dimension.  Values should be sorted and can be repeated.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Has same shape as data, except for dimension 0 which
// has size `k`, the number of segments.
Node* SegmentMin(NodeOut data, NodeOut segment_ids, const
                 GraphDefBuilder::Options& opts);

// Computes the product along segments of a tensor.
//
// Read [the section on
// Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
// of segments.
//
// Computes a tensor such that
// \\(output_i = \prod_j data_j\\) where the product is over `j` such
// that `segment_ids[j] == i`.
//
// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
// <img style="width:100%" src="../../images/SegmentProd.png" alt>
// </div>
//
// Arguments:
// * segment_ids: A 1-D tensor whose rank is equal to the rank of `data`'s
// first dimension.  Values should be sorted and can be repeated.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Has same shape as data, except for dimension 0 which
// has size `k`, the number of segments.
Node* SegmentProd(NodeOut data, NodeOut segment_ids, const
                  GraphDefBuilder::Options& opts);

// Computes the sum along segments of a tensor.
//
// Read [the section on Segmentation](../../api_docs/python/math_ops.md#segmentation)
// for an explanation of segments.
//
// Computes a tensor such that
// \\(output_i = \sum_j data_j\\) where sum is over `j` such
// that `segment_ids[j] == i`.
//
// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
// <img style="width:100%" src="../../images/SegmentSum.png" alt>
// </div>
//
// Arguments:
// * segment_ids: A 1-D tensor whose rank is equal to the rank of `data`'s
// first dimension.  Values should be sorted and can be repeated.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Has same shape as data, except for dimension 0 which
// has size `k`, the number of segments.
Node* SegmentSum(NodeOut data, NodeOut segment_ids, const
                 GraphDefBuilder::Options& opts);

// Selects elements from `t` or `e`, depending on `condition`.
//
// The `condition`, `t`, and `e` tensors must all have the same shape,
// and the output will also have that shape. The `condition` tensor acts
// as an element-wise mask that chooses, based on the value at each
// element, whether the corresponding element in the output should be
// taken from `t` (if true) or `e` (if false). For example:
//
// For example:
//
// ```prettyprint
// # 'condition' tensor is [[True, False]
// #                        [True, False]]
// # 't' is [[1, 1],
// #         [1, 1]]
// # 'e' is [[2, 2],
// #         [2, 2]]
// select(condition, t, e) ==> [[1, 2],
//                              [1, 2]]
// ```
//
// Arguments:
// * t: = A `Tensor` with the same shape as `condition`.
// * e: = A `Tensor` with the same type and shape as `t`.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// A `Tensor` with the same type and shape as `t` and `e`.
Node* Select(NodeOut condition, NodeOut t, NodeOut e, const
             GraphDefBuilder::Options& opts);

// Computes sigmoid of `x` element-wise.
//
// Specifically, `y = 1 / (1 + exp(-x))`.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* Sigmoid(NodeOut x, const GraphDefBuilder::Options& opts);

// Returns an element-wise indication of the sign of a number.
//
// y = sign(x) = -1 if x < 0; 0 if x == 0; 1 if x > 0.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* Sign(NodeOut x, const GraphDefBuilder::Options& opts);

// Computes sin of x element-wise.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* Sin(NodeOut x, const GraphDefBuilder::Options& opts);

// Multiply matrix "a" by matrix "b".
//
// The inputs must be two-dimensional matrices and the inner dimension of "a" must
// match the outer dimension of "b". This op is optimized for the case where at
// least one of "a" or "b" is sparse. The breakeven for using this versus a dense
// matrix multiply on one platform was 30% zero values in the sparse matrix.
//
// Arguments:
// * opts:
//   .WithAttr("transpose_a", bool): Defaults to false.
//   .WithAttr("transpose_b", bool): Defaults to false.
//   .WithAttr("a_is_sparse", bool): Defaults to false.
//   .WithAttr("b_is_sparse", bool): Defaults to false.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* SparseMatMul(NodeOut a, NodeOut b, const GraphDefBuilder::Options& opts);

// Computes the mean along sparse segments of a tensor.
//
// Read [the section on
// Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
// of segments.
//
// Like `SegmentMean`, but `segment_ids` can have rank less than `data`'s first
// dimension, selecting a subset of dimension 0, specified by `indices`.
//
// Arguments:
// * indices: A 1-D tensor. Has same rank as `segment_ids`.
// * segment_ids: A 1-D tensor. Values should be sorted and can be repeated.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Has same shape as data, except for dimension 0 which
// has size `k`, the number of segments.
Node* SparseSegmentMean(NodeOut data, NodeOut indices, NodeOut segment_ids,
                        const GraphDefBuilder::Options& opts);

// Computes gradients for SparseSegmentMean.
//
// Returns tensor "output" with same shape as grad, except for dimension 0 whose
// value is output_dim0.
//
// Arguments:
// * grad: gradient propagated to the SparseSegmentMean op.
// * indices: indices passed to the corresponding SparseSegmentMean op.
// * segment_ids: segment_ids passed to the corresponding SparseSegmentMean op.
// * output_dim0: dimension 0 of "data" passed to SparseSegmentMean op.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* SparseSegmentMeanGrad(NodeOut grad, NodeOut indices, NodeOut segment_ids,
                            NodeOut output_dim0, const
                            GraphDefBuilder::Options& opts);

// Computes the sum along sparse segments of a tensor divided by the sqrt of N.
//
// N is the size of the segment being reduced.
//
// Read [the section on
// Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
// of segments.
//
// Arguments:
// * indices: A 1-D tensor. Has same rank as `segment_ids`.
// * segment_ids: A 1-D tensor. Values should be sorted and can be repeated.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Has same shape as data, except for dimension 0 which
// has size `k`, the number of segments.
Node* SparseSegmentSqrtN(NodeOut data, NodeOut indices, NodeOut segment_ids,
                         const GraphDefBuilder::Options& opts);

// Computes gradients for SparseSegmentSqrtN.
//
// Returns tensor "output" with same shape as grad, except for dimension 0 whose
// value is output_dim0.
//
// Arguments:
// * grad: gradient propagated to the SparseSegmentSqrtN op.
// * indices: indices passed to the corresponding SparseSegmentSqrtN op.
// * segment_ids: segment_ids passed to the corresponding SparseSegmentSqrtN op.
// * output_dim0: dimension 0 of "data" passed to SparseSegmentSqrtN op.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* SparseSegmentSqrtNGrad(NodeOut grad, NodeOut indices, NodeOut
                             segment_ids, NodeOut output_dim0, const
                             GraphDefBuilder::Options& opts);

// Computes the sum along sparse segments of a tensor.
//
// Read [the section on
// Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
// of segments.
//
// Like `SegmentSum`, but `segment_ids` can have rank less than `data`'s first
// dimension, selecting a subset of dimension 0, specified by `indices`.
//
// For example:
//
// ```prettyprint
// c = tf.constant([[1,2,3,4], [-1,-2,-3,-4], [5,6,7,8]])
//
// # Select two rows, one segment.
// tf.sparse_segment_sum(c, tf.constant([0, 1]), tf.constant([0, 0]))
//   ==> [[0 0 0 0]]
//
// # Select two rows, two segment.
// tf.sparse_segment_sum(c, tf.constant([0, 1]), tf.constant([0, 1]))
//   ==> [[ 1  2  3  4]
//        [-1 -2 -3 -4]]
//
// # Select all rows, two segments.
// tf.sparse_segment_sum(c, tf.constant([0, 1, 2]), tf.constant([0, 0, 1]))
//   ==> [[0 0 0 0]
//        [5 6 7 8]]
//
// # Which is equivalent to:
// tf.segment_sum(c, tf.constant([0, 0, 1]))
// ```
//
// Arguments:
// * indices: A 1-D tensor. Has same rank as `segment_ids`.
// * segment_ids: A 1-D tensor. Values should be sorted and can be repeated.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Has same shape as data, except for dimension 0 which
// has size `k`, the number of segments.
Node* SparseSegmentSum(NodeOut data, NodeOut indices, NodeOut segment_ids,
                       const GraphDefBuilder::Options& opts);

// Computes square root of x element-wise.
//
// I.e., \\(y = \sqrt{x} = x^{1/2}\\).
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* Sqrt(NodeOut x, const GraphDefBuilder::Options& opts);

// Computes square of x element-wise.
//
// I.e., \\(y = x * x = x^2\\).
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* Square(NodeOut x, const GraphDefBuilder::Options& opts);

// Returns x - y element-wise.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* Sub(NodeOut x, NodeOut y, const GraphDefBuilder::Options& opts);

// Computes the sum of elements across dimensions of a tensor.
//
// Reduces `input` along the dimensions given in `reduction_indices`. Unless
// `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
// `reduction_indices`. If `keep_dims` is true, the reduced dimensions are
// retained with length 1.
//
// Arguments:
// * input: The tensor to reduce.
// * reduction_indices: The dimensions to reduce.
// * opts:
//   .WithAttr("keep_dims", bool): Defaults to false.
//     If true, retain reduced dimensions with length 1.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The reduced tensor.
Node* Sum(NodeOut input, NodeOut reduction_indices, const
          GraphDefBuilder::Options& opts);

// Computes hyperbolic tangent of `x` element-wise.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* Tanh(NodeOut x, const GraphDefBuilder::Options& opts);

// Computes the sum along segments of a tensor.
//
// Read [the section on
// Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
// of segments.
//
// Computes a tensor such that
// \\(output_i = \sum_j data_j\\) where sum is over `j` such
// that `segment_ids[j] == i`. Unlike `SegmentSum`, `segment_ids`
// need not be sorted and need not cover all values in the full
//   range of valid values.
//
// If the sum is empty for a given segment ID `i`, `output[i] = 0`.
//
// `num_segments` should equal the number of distinct segment IDs.
//
// <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
// <img style="width:100%" src="../../images/UnsortedSegmentSum.png" alt>
// </div>
//
// Arguments:
// * segment_ids: A 1-D tensor whose rank is equal to the rank of `data`'s
// first dimension.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Has same shape as data, except for dimension 0 which
// has size `num_segments`.
Node* UnsortedSegmentSum(NodeOut data, NodeOut segment_ids, NodeOut
                         num_segments, const GraphDefBuilder::Options& opts);

}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_OPS_MATH_OPS_H_
