// This file is MACHINE GENERATED! Do not edit.

#include "tensorflow/cc/ops/summary_ops.h"

#include "tensorflow/core/graph/node_builder.h"

namespace tensorflow {
namespace ops {

Node* HistogramSummary(NodeOut tag, NodeOut values, const
                       GraphDefBuilder::Options& opts) {
  static const string kOpName = "HistogramSummary";
  return BinaryOp(kOpName, tag, values, opts);
}

Node* ImageSummary(NodeOut tag, NodeOut tensor, const GraphDefBuilder::Options&
                   opts) {
  static const string kOpName = "ImageSummary";
  return BinaryOp(kOpName, tag, tensor, opts);
}

Node* MergeSummary(gtl::ArraySlice<NodeOut> inputs, const
                   GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "MergeSummary";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(inputs);
  return opts.FinalizeBuilder(&node_builder);
}

Node* ScalarSummary(NodeOut tags, NodeOut values, const
                    GraphDefBuilder::Options& opts) {
  static const string kOpName = "ScalarSummary";
  return BinaryOp(kOpName, tags, values, opts);
}

}  // namespace ops
}  // namespace tensorflow
