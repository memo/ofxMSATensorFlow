// This file is MACHINE GENERATED! Do not edit.

#include "tensorflow/cc/ops/string_ops.h"

#include "tensorflow/core/graph/node_builder.h"

namespace tensorflow {
namespace ops {

Node* StringToHashBucket(NodeOut string_tensor, int64 num_buckets, const
                         GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "StringToHashBucket";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(string_tensor);
  node_builder.Attr("num_buckets", num_buckets);
  return opts.FinalizeBuilder(&node_builder);
}

}  // namespace ops
}  // namespace tensorflow
