// This file is MACHINE GENERATED! Do not edit.

#include "tensorflow/cc/ops/attention_ops.h"

#include "tensorflow/core/graph/node_builder.h"

namespace tensorflow {
namespace ops {

Node* ExtractGlimpse(NodeOut input, NodeOut size, NodeOut offsets, const
                     GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "ExtractGlimpse";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(input);
  node_builder.Input(size);
  node_builder.Input(offsets);
  return opts.FinalizeBuilder(&node_builder);
}

}  // namespace ops
}  // namespace tensorflow
