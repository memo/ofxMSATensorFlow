// This file is MACHINE GENERATED! Do not edit.

#include "tensorflow/cc/ops/sparse_ops.h"

#include "tensorflow/core/graph/node_builder.h"

namespace tensorflow {
namespace ops {

Node* DeserializeManySparse(NodeOut serialized_sparse, DataType dtype, const
                            GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "DeserializeManySparse";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(serialized_sparse);
  node_builder.Attr("dtype", dtype);
  return opts.FinalizeBuilder(&node_builder);
}

Node* SerializeManySparse(NodeOut sparse_indices, NodeOut sparse_values,
                          NodeOut sparse_shape, const GraphDefBuilder::Options&
                          opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "SerializeManySparse";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(sparse_indices);
  node_builder.Input(sparse_values);
  node_builder.Input(sparse_shape);
  return opts.FinalizeBuilder(&node_builder);
}

Node* SerializeSparse(NodeOut sparse_indices, NodeOut sparse_values, NodeOut
                      sparse_shape, const GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "SerializeSparse";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(sparse_indices);
  node_builder.Input(sparse_values);
  node_builder.Input(sparse_shape);
  return opts.FinalizeBuilder(&node_builder);
}

Node* SparseConcat(gtl::ArraySlice<NodeOut> indices, gtl::ArraySlice<NodeOut>
                   values, gtl::ArraySlice<NodeOut> shapes, int64 concat_dim,
                   const GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "SparseConcat";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(indices);
  node_builder.Input(values);
  node_builder.Input(shapes);
  node_builder.Attr("concat_dim", concat_dim);
  return opts.FinalizeBuilder(&node_builder);
}

Node* SparseReorder(NodeOut input_indices, NodeOut input_values, NodeOut
                    input_shape, const GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "SparseReorder";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(input_indices);
  node_builder.Input(input_values);
  node_builder.Input(input_shape);
  return opts.FinalizeBuilder(&node_builder);
}

Node* SparseSplit(NodeOut split_dim, NodeOut indices, NodeOut values, NodeOut
                  shape, int64 num_split, const GraphDefBuilder::Options& opts)
                  {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "SparseSplit";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(split_dim);
  node_builder.Input(indices);
  node_builder.Input(values);
  node_builder.Input(shape);
  node_builder.Attr("num_split", num_split);
  return opts.FinalizeBuilder(&node_builder);
}

Node* SparseTensorDenseMatMul(NodeOut a_indices, NodeOut a_values, NodeOut
                              a_shape, NodeOut b, const
                              GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "SparseTensorDenseMatMul";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(a_indices);
  node_builder.Input(a_values);
  node_builder.Input(a_shape);
  node_builder.Input(b);
  return opts.FinalizeBuilder(&node_builder);
}

Node* SparseToDense(NodeOut sparse_indices, NodeOut output_shape, NodeOut
                    sparse_values, NodeOut default_value, const
                    GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "SparseToDense";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(sparse_indices);
  node_builder.Input(output_shape);
  node_builder.Input(sparse_values);
  node_builder.Input(default_value);
  return opts.FinalizeBuilder(&node_builder);
}

}  // namespace ops
}  // namespace tensorflow
