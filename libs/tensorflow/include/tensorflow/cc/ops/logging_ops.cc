// This file is MACHINE GENERATED! Do not edit.

#include "tensorflow/cc/ops/logging_ops.h"

#include "tensorflow/core/graph/node_builder.h"

namespace tensorflow {
namespace ops {

Node* Assert(NodeOut condition, gtl::ArraySlice<NodeOut> data, const
             GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "Assert";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(condition);
  node_builder.Input(data);
  return opts.FinalizeBuilder(&node_builder);
}

Node* Print(NodeOut input, gtl::ArraySlice<NodeOut> data, const
            GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "Print";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(input);
  node_builder.Input(data);
  return opts.FinalizeBuilder(&node_builder);
}

}  // namespace ops
}  // namespace tensorflow
