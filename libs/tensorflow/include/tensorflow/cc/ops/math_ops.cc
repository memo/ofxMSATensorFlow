// This file is MACHINE GENERATED! Do not edit.

#include "tensorflow/cc/ops/math_ops.h"

#include "tensorflow/core/graph/node_builder.h"

namespace tensorflow {
namespace ops {

Node* Abs(NodeOut x, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "Abs";
  return UnaryOp(kOpName, x, opts);
}

Node* Add(NodeOut x, NodeOut y, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "Add";
  return BinaryOp(kOpName, x, y, opts);
}

Node* AddN(gtl::ArraySlice<NodeOut> inputs, const GraphDefBuilder::Options&
           opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "AddN";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(inputs);
  return opts.FinalizeBuilder(&node_builder);
}

Node* All(NodeOut input, NodeOut reduction_indices, const
          GraphDefBuilder::Options& opts) {
  static const string kOpName = "All";
  return BinaryOp(kOpName, input, reduction_indices, opts);
}

Node* Any(NodeOut input, NodeOut reduction_indices, const
          GraphDefBuilder::Options& opts) {
  static const string kOpName = "Any";
  return BinaryOp(kOpName, input, reduction_indices, opts);
}

Node* ArgMax(NodeOut input, NodeOut dimension, const GraphDefBuilder::Options&
             opts) {
  static const string kOpName = "ArgMax";
  return BinaryOp(kOpName, input, dimension, opts);
}

Node* ArgMin(NodeOut input, NodeOut dimension, const GraphDefBuilder::Options&
             opts) {
  static const string kOpName = "ArgMin";
  return BinaryOp(kOpName, input, dimension, opts);
}

Node* BatchMatMul(NodeOut x, NodeOut y, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "BatchMatMul";
  return BinaryOp(kOpName, x, y, opts);
}

Node* Cast(NodeOut x, DataType DstT, const GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "Cast";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(x);
  node_builder.Attr("DstT", DstT);
  return opts.FinalizeBuilder(&node_builder);
}

Node* Ceil(NodeOut x, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "Ceil";
  return UnaryOp(kOpName, x, opts);
}

Node* Complex(NodeOut real, NodeOut imag, const GraphDefBuilder::Options& opts)
              {
  static const string kOpName = "Complex";
  return BinaryOp(kOpName, real, imag, opts);
}

Node* ComplexAbs(NodeOut x, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "ComplexAbs";
  return UnaryOp(kOpName, x, opts);
}

Node* Conj(NodeOut in, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "Conj";
  return UnaryOp(kOpName, in, opts);
}

Node* Cos(NodeOut x, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "Cos";
  return UnaryOp(kOpName, x, opts);
}

Node* Div(NodeOut x, NodeOut y, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "Div";
  return BinaryOp(kOpName, x, y, opts);
}

Node* Equal(NodeOut x, NodeOut y, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "Equal";
  return BinaryOp(kOpName, x, y, opts);
}

Node* Erf(NodeOut x, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "Erf";
  return UnaryOp(kOpName, x, opts);
}

Node* Erfc(NodeOut x, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "Erfc";
  return UnaryOp(kOpName, x, opts);
}

Node* Exp(NodeOut x, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "Exp";
  return UnaryOp(kOpName, x, opts);
}

Node* FFT2D(NodeOut in, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "FFT2D";
  return UnaryOp(kOpName, in, opts);
}

Node* Floor(NodeOut x, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "Floor";
  return UnaryOp(kOpName, x, opts);
}

Node* Greater(NodeOut x, NodeOut y, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "Greater";
  return BinaryOp(kOpName, x, y, opts);
}

Node* GreaterEqual(NodeOut x, NodeOut y, const GraphDefBuilder::Options& opts)
                   {
  static const string kOpName = "GreaterEqual";
  return BinaryOp(kOpName, x, y, opts);
}

Node* IFFT2D(NodeOut in, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "IFFT2D";
  return UnaryOp(kOpName, in, opts);
}

Node* Imag(NodeOut in, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "Imag";
  return UnaryOp(kOpName, in, opts);
}

Node* Inv(NodeOut x, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "Inv";
  return UnaryOp(kOpName, x, opts);
}

Node* IsFinite(NodeOut x, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "IsFinite";
  return UnaryOp(kOpName, x, opts);
}

Node* IsInf(NodeOut x, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "IsInf";
  return UnaryOp(kOpName, x, opts);
}

Node* IsNan(NodeOut x, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "IsNan";
  return UnaryOp(kOpName, x, opts);
}

Node* Less(NodeOut x, NodeOut y, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "Less";
  return BinaryOp(kOpName, x, y, opts);
}

Node* LessEqual(NodeOut x, NodeOut y, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "LessEqual";
  return BinaryOp(kOpName, x, y, opts);
}

Node* Lgamma(NodeOut x, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "Lgamma";
  return UnaryOp(kOpName, x, opts);
}

Node* LinSpace(NodeOut start, NodeOut stop, NodeOut num, const
               GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "LinSpace";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(start);
  node_builder.Input(stop);
  node_builder.Input(num);
  return opts.FinalizeBuilder(&node_builder);
}

Node* Log(NodeOut x, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "Log";
  return UnaryOp(kOpName, x, opts);
}

Node* LogicalAnd(NodeOut x, NodeOut y, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "LogicalAnd";
  return BinaryOp(kOpName, x, y, opts);
}

Node* LogicalNot(NodeOut x, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "LogicalNot";
  return UnaryOp(kOpName, x, opts);
}

Node* LogicalOr(NodeOut x, NodeOut y, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "LogicalOr";
  return BinaryOp(kOpName, x, y, opts);
}

Node* MatMul(NodeOut a, NodeOut b, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "MatMul";
  return BinaryOp(kOpName, a, b, opts);
}

Node* Max(NodeOut input, NodeOut reduction_indices, const
          GraphDefBuilder::Options& opts) {
  static const string kOpName = "Max";
  return BinaryOp(kOpName, input, reduction_indices, opts);
}

Node* Maximum(NodeOut x, NodeOut y, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "Maximum";
  return BinaryOp(kOpName, x, y, opts);
}

Node* Mean(NodeOut input, NodeOut reduction_indices, const
           GraphDefBuilder::Options& opts) {
  static const string kOpName = "Mean";
  return BinaryOp(kOpName, input, reduction_indices, opts);
}

Node* Min(NodeOut input, NodeOut reduction_indices, const
          GraphDefBuilder::Options& opts) {
  static const string kOpName = "Min";
  return BinaryOp(kOpName, input, reduction_indices, opts);
}

Node* Minimum(NodeOut x, NodeOut y, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "Minimum";
  return BinaryOp(kOpName, x, y, opts);
}

Node* Mod(NodeOut x, NodeOut y, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "Mod";
  return BinaryOp(kOpName, x, y, opts);
}

Node* Mul(NodeOut x, NodeOut y, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "Mul";
  return BinaryOp(kOpName, x, y, opts);
}

Node* Neg(NodeOut x, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "Neg";
  return UnaryOp(kOpName, x, opts);
}

Node* NotEqual(NodeOut x, NodeOut y, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "NotEqual";
  return BinaryOp(kOpName, x, y, opts);
}

Node* Pow(NodeOut x, NodeOut y, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "Pow";
  return BinaryOp(kOpName, x, y, opts);
}

Node* Prod(NodeOut input, NodeOut reduction_indices, const
           GraphDefBuilder::Options& opts) {
  static const string kOpName = "Prod";
  return BinaryOp(kOpName, input, reduction_indices, opts);
}

Node* Range(NodeOut start, NodeOut limit, NodeOut delta, const
            GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "Range";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(start);
  node_builder.Input(limit);
  node_builder.Input(delta);
  return opts.FinalizeBuilder(&node_builder);
}

Node* Real(NodeOut in, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "Real";
  return UnaryOp(kOpName, in, opts);
}

Node* Rsqrt(NodeOut x, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "Rsqrt";
  return UnaryOp(kOpName, x, opts);
}

Node* SegmentMax(NodeOut data, NodeOut segment_ids, const
                 GraphDefBuilder::Options& opts) {
  static const string kOpName = "SegmentMax";
  return BinaryOp(kOpName, data, segment_ids, opts);
}

Node* SegmentMean(NodeOut data, NodeOut segment_ids, const
                  GraphDefBuilder::Options& opts) {
  static const string kOpName = "SegmentMean";
  return BinaryOp(kOpName, data, segment_ids, opts);
}

Node* SegmentMin(NodeOut data, NodeOut segment_ids, const
                 GraphDefBuilder::Options& opts) {
  static const string kOpName = "SegmentMin";
  return BinaryOp(kOpName, data, segment_ids, opts);
}

Node* SegmentProd(NodeOut data, NodeOut segment_ids, const
                  GraphDefBuilder::Options& opts) {
  static const string kOpName = "SegmentProd";
  return BinaryOp(kOpName, data, segment_ids, opts);
}

Node* SegmentSum(NodeOut data, NodeOut segment_ids, const
                 GraphDefBuilder::Options& opts) {
  static const string kOpName = "SegmentSum";
  return BinaryOp(kOpName, data, segment_ids, opts);
}

Node* Select(NodeOut condition, NodeOut t, NodeOut e, const
             GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "Select";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(condition);
  node_builder.Input(t);
  node_builder.Input(e);
  return opts.FinalizeBuilder(&node_builder);
}

Node* Sigmoid(NodeOut x, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "Sigmoid";
  return UnaryOp(kOpName, x, opts);
}

Node* Sign(NodeOut x, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "Sign";
  return UnaryOp(kOpName, x, opts);
}

Node* Sin(NodeOut x, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "Sin";
  return UnaryOp(kOpName, x, opts);
}

Node* SparseMatMul(NodeOut a, NodeOut b, const GraphDefBuilder::Options& opts)
                   {
  static const string kOpName = "SparseMatMul";
  return BinaryOp(kOpName, a, b, opts);
}

Node* SparseSegmentMean(NodeOut data, NodeOut indices, NodeOut segment_ids,
                        const GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "SparseSegmentMean";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(data);
  node_builder.Input(indices);
  node_builder.Input(segment_ids);
  return opts.FinalizeBuilder(&node_builder);
}

Node* SparseSegmentMeanGrad(NodeOut grad, NodeOut indices, NodeOut segment_ids,
                            NodeOut output_dim0, const
                            GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "SparseSegmentMeanGrad";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(grad);
  node_builder.Input(indices);
  node_builder.Input(segment_ids);
  node_builder.Input(output_dim0);
  return opts.FinalizeBuilder(&node_builder);
}

Node* SparseSegmentSqrtN(NodeOut data, NodeOut indices, NodeOut segment_ids,
                         const GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "SparseSegmentSqrtN";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(data);
  node_builder.Input(indices);
  node_builder.Input(segment_ids);
  return opts.FinalizeBuilder(&node_builder);
}

Node* SparseSegmentSqrtNGrad(NodeOut grad, NodeOut indices, NodeOut
                             segment_ids, NodeOut output_dim0, const
                             GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "SparseSegmentSqrtNGrad";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(grad);
  node_builder.Input(indices);
  node_builder.Input(segment_ids);
  node_builder.Input(output_dim0);
  return opts.FinalizeBuilder(&node_builder);
}

Node* SparseSegmentSum(NodeOut data, NodeOut indices, NodeOut segment_ids,
                       const GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "SparseSegmentSum";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(data);
  node_builder.Input(indices);
  node_builder.Input(segment_ids);
  return opts.FinalizeBuilder(&node_builder);
}

Node* Sqrt(NodeOut x, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "Sqrt";
  return UnaryOp(kOpName, x, opts);
}

Node* Square(NodeOut x, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "Square";
  return UnaryOp(kOpName, x, opts);
}

Node* Sub(NodeOut x, NodeOut y, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "Sub";
  return BinaryOp(kOpName, x, y, opts);
}

Node* Sum(NodeOut input, NodeOut reduction_indices, const
          GraphDefBuilder::Options& opts) {
  static const string kOpName = "Sum";
  return BinaryOp(kOpName, input, reduction_indices, opts);
}

Node* Tanh(NodeOut x, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "Tanh";
  return UnaryOp(kOpName, x, opts);
}

Node* UnsortedSegmentSum(NodeOut data, NodeOut segment_ids, NodeOut
                         num_segments, const GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "UnsortedSegmentSum";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(data);
  node_builder.Input(segment_ids);
  node_builder.Input(num_segments);
  return opts.FinalizeBuilder(&node_builder);
}

}  // namespace ops
}  // namespace tensorflow
