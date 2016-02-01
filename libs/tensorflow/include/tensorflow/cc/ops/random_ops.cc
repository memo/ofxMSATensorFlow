// This file is MACHINE GENERATED! Do not edit.

#include "tensorflow/cc/ops/random_ops.h"

#include "tensorflow/core/graph/node_builder.h"

namespace tensorflow {
namespace ops {

Node* RandomShuffle(NodeOut value, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "RandomShuffle";
  return UnaryOp(kOpName, value, opts);
}

Node* RandomStandardNormal(NodeOut shape, DataType dtype, const
                           GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "RandomStandardNormal";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(shape);
  node_builder.Attr("dtype", dtype);
  return opts.FinalizeBuilder(&node_builder);
}

Node* RandomUniform(NodeOut shape, DataType dtype, const
                    GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "RandomUniform";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(shape);
  node_builder.Attr("dtype", dtype);
  return opts.FinalizeBuilder(&node_builder);
}

Node* RandomUniformInt(NodeOut shape, NodeOut minval, NodeOut maxval, const
                       GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "RandomUniformInt";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(shape);
  node_builder.Input(minval);
  node_builder.Input(maxval);
  return opts.FinalizeBuilder(&node_builder);
}

Node* TruncatedNormal(NodeOut shape, DataType dtype, const
                      GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "TruncatedNormal";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(shape);
  node_builder.Attr("dtype", dtype);
  return opts.FinalizeBuilder(&node_builder);
}

}  // namespace ops
}  // namespace tensorflow
