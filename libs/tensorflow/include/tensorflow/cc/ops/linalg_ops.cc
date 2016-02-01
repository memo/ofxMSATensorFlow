// This file is MACHINE GENERATED! Do not edit.

#include "tensorflow/cc/ops/linalg_ops.h"

#include "tensorflow/core/graph/node_builder.h"

namespace tensorflow {
namespace ops {

Node* BatchCholesky(NodeOut input, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "BatchCholesky";
  return UnaryOp(kOpName, input, opts);
}

Node* BatchMatrixDeterminant(NodeOut input, const GraphDefBuilder::Options&
                             opts) {
  static const string kOpName = "BatchMatrixDeterminant";
  return UnaryOp(kOpName, input, opts);
}

Node* BatchMatrixInverse(NodeOut input, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "BatchMatrixInverse";
  return UnaryOp(kOpName, input, opts);
}

Node* BatchMatrixSolve(NodeOut matrix, NodeOut rhs, const
                       GraphDefBuilder::Options& opts) {
  static const string kOpName = "BatchMatrixSolve";
  return BinaryOp(kOpName, matrix, rhs, opts);
}

Node* BatchMatrixSolveLs(NodeOut matrix, NodeOut rhs, NodeOut l2_regularizer,
                         const GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "BatchMatrixSolveLs";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(matrix);
  node_builder.Input(rhs);
  node_builder.Input(l2_regularizer);
  return opts.FinalizeBuilder(&node_builder);
}

Node* BatchMatrixTriangularSolve(NodeOut matrix, NodeOut rhs, const
                                 GraphDefBuilder::Options& opts) {
  static const string kOpName = "BatchMatrixTriangularSolve";
  return BinaryOp(kOpName, matrix, rhs, opts);
}

Node* BatchSelfAdjointEig(NodeOut input, const GraphDefBuilder::Options& opts)
                          {
  static const string kOpName = "BatchSelfAdjointEig";
  return UnaryOp(kOpName, input, opts);
}

Node* Cholesky(NodeOut input, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "Cholesky";
  return UnaryOp(kOpName, input, opts);
}

Node* MatrixDeterminant(NodeOut input, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "MatrixDeterminant";
  return UnaryOp(kOpName, input, opts);
}

Node* MatrixInverse(NodeOut input, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "MatrixInverse";
  return UnaryOp(kOpName, input, opts);
}

Node* MatrixSolve(NodeOut matrix, NodeOut rhs, const GraphDefBuilder::Options&
                  opts) {
  static const string kOpName = "MatrixSolve";
  return BinaryOp(kOpName, matrix, rhs, opts);
}

Node* MatrixSolveLs(NodeOut matrix, NodeOut rhs, NodeOut l2_regularizer, const
                    GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "MatrixSolveLs";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(matrix);
  node_builder.Input(rhs);
  node_builder.Input(l2_regularizer);
  return opts.FinalizeBuilder(&node_builder);
}

Node* MatrixTriangularSolve(NodeOut matrix, NodeOut rhs, const
                            GraphDefBuilder::Options& opts) {
  static const string kOpName = "MatrixTriangularSolve";
  return BinaryOp(kOpName, matrix, rhs, opts);
}

Node* SelfAdjointEig(NodeOut input, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "SelfAdjointEig";
  return UnaryOp(kOpName, input, opts);
}

}  // namespace ops
}  // namespace tensorflow
