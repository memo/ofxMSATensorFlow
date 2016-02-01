// This file is MACHINE GENERATED! Do not edit.

#include "tensorflow/cc/ops/parsing_ops.h"

#include "tensorflow/core/graph/node_builder.h"

namespace tensorflow {
namespace ops {

Node* DecodeCSV(NodeOut records, gtl::ArraySlice<NodeOut> record_defaults,
                const GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "DecodeCSV";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(records);
  node_builder.Input(record_defaults);
  return opts.FinalizeBuilder(&node_builder);
}

Node* DecodeJSONExample(NodeOut json_examples, const GraphDefBuilder::Options&
                        opts) {
  static const string kOpName = "DecodeJSONExample";
  return UnaryOp(kOpName, json_examples, opts);
}

Node* DecodeRaw(NodeOut bytes, DataType out_type, const
                GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "DecodeRaw";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(bytes);
  node_builder.Attr("out_type", out_type);
  return opts.FinalizeBuilder(&node_builder);
}

Node* ParseExample(NodeOut serialized, NodeOut names, gtl::ArraySlice<NodeOut>
                   sparse_keys, gtl::ArraySlice<NodeOut> dense_keys,
                   gtl::ArraySlice<NodeOut> dense_defaults, DataTypeSlice
                   sparse_types, gtl::ArraySlice<TensorShape> dense_shapes,
                   const GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "ParseExample";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(serialized);
  node_builder.Input(names);
  node_builder.Input(sparse_keys);
  node_builder.Input(dense_keys);
  node_builder.Input(dense_defaults);
  node_builder.Attr("sparse_types", sparse_types);
  node_builder.Attr("dense_shapes", dense_shapes);
  return opts.FinalizeBuilder(&node_builder);
}

Node* ParseSingleSequenceExample(NodeOut serialized, NodeOut
                                 feature_list_dense_missing_assumed_empty,
                                 gtl::ArraySlice<NodeOut> context_sparse_keys,
                                 gtl::ArraySlice<NodeOut> context_dense_keys,
                                 gtl::ArraySlice<NodeOut>
                                 feature_list_sparse_keys,
                                 gtl::ArraySlice<NodeOut>
                                 feature_list_dense_keys,
                                 gtl::ArraySlice<NodeOut>
                                 context_dense_defaults, NodeOut debug_name,
                                 const GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "ParseSingleSequenceExample";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(serialized);
  node_builder.Input(feature_list_dense_missing_assumed_empty);
  node_builder.Input(context_sparse_keys);
  node_builder.Input(context_dense_keys);
  node_builder.Input(feature_list_sparse_keys);
  node_builder.Input(feature_list_dense_keys);
  node_builder.Input(context_dense_defaults);
  node_builder.Input(debug_name);
  return opts.FinalizeBuilder(&node_builder);
}

Node* StringToNumber(NodeOut string_tensor, const GraphDefBuilder::Options&
                     opts) {
  static const string kOpName = "StringToNumber";
  return UnaryOp(kOpName, string_tensor, opts);
}

}  // namespace ops
}  // namespace tensorflow
