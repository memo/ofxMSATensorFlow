// This file is MACHINE GENERATED! Do not edit.

#include "tensorflow/cc/ops/data_flow_ops.h"

#include "tensorflow/core/graph/node_builder.h"

namespace tensorflow {
namespace ops {

Node* DynamicPartition(NodeOut data, NodeOut partitions, int64 num_partitions,
                       const GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "DynamicPartition";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(data);
  node_builder.Input(partitions);
  node_builder.Attr("num_partitions", num_partitions);
  return opts.FinalizeBuilder(&node_builder);
}

Node* DynamicStitch(gtl::ArraySlice<NodeOut> indices, gtl::ArraySlice<NodeOut>
                    data, const GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "DynamicStitch";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(indices);
  node_builder.Input(data);
  return opts.FinalizeBuilder(&node_builder);
}

Node* FIFOQueue(DataTypeSlice component_types, const GraphDefBuilder::Options&
                opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "FIFOQueue";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Attr("component_types", component_types);
  return opts.FinalizeBuilder(&node_builder);
}

Node* HashTable(DataType key_dtype, DataType value_dtype, const
                GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "HashTable";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Attr("key_dtype", key_dtype);
  node_builder.Attr("value_dtype", value_dtype);
  return opts.FinalizeBuilder(&node_builder);
}

Node* InitializeTable(NodeOut table_handle, NodeOut keys, NodeOut values, const
                      GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "InitializeTable";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(table_handle);
  node_builder.Input(keys);
  node_builder.Input(values);
  return opts.FinalizeBuilder(&node_builder);
}

Node* LookupTableFind(NodeOut table_handle, NodeOut keys, NodeOut
                      default_value, const GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "LookupTableFind";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(table_handle);
  node_builder.Input(keys);
  node_builder.Input(default_value);
  return opts.FinalizeBuilder(&node_builder);
}

Node* LookupTableSize(NodeOut table_handle, const GraphDefBuilder::Options&
                      opts) {
  static const string kOpName = "LookupTableSize";
  return UnaryOp(kOpName, table_handle, opts);
}

Node* PaddingFIFOQueue(DataTypeSlice component_types, const
                       GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "PaddingFIFOQueue";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Attr("component_types", component_types);
  return opts.FinalizeBuilder(&node_builder);
}

Node* QueueClose(NodeOut handle, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "QueueClose";
  return UnaryOp(kOpName, handle, opts);
}

Node* QueueDequeue(NodeOut handle, DataTypeSlice component_types, const
                   GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "QueueDequeue";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(handle);
  node_builder.Attr("component_types", component_types);
  return opts.FinalizeBuilder(&node_builder);
}

Node* QueueDequeueMany(NodeOut handle, NodeOut n, DataTypeSlice
                       component_types, const GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "QueueDequeueMany";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(handle);
  node_builder.Input(n);
  node_builder.Attr("component_types", component_types);
  return opts.FinalizeBuilder(&node_builder);
}

Node* QueueEnqueue(NodeOut handle, gtl::ArraySlice<NodeOut> components, const
                   GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "QueueEnqueue";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(handle);
  node_builder.Input(components);
  return opts.FinalizeBuilder(&node_builder);
}

Node* QueueEnqueueMany(NodeOut handle, gtl::ArraySlice<NodeOut> components,
                       const GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "QueueEnqueueMany";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(handle);
  node_builder.Input(components);
  return opts.FinalizeBuilder(&node_builder);
}

Node* QueueSize(NodeOut handle, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "QueueSize";
  return UnaryOp(kOpName, handle, opts);
}

Node* RandomShuffleQueue(DataTypeSlice component_types, const
                         GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "RandomShuffleQueue";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Attr("component_types", component_types);
  return opts.FinalizeBuilder(&node_builder);
}

Node* Stack(DataType elem_type, const GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "Stack";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Attr("elem_type", elem_type);
  return opts.FinalizeBuilder(&node_builder);
}

Node* StackClose(NodeOut handle, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "StackClose";
  return UnaryOp(kOpName, handle, opts);
}

Node* StackPop(NodeOut handle, DataType elem_type, const
               GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "StackPop";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(handle);
  node_builder.Attr("elem_type", elem_type);
  return opts.FinalizeBuilder(&node_builder);
}

Node* StackPush(NodeOut handle, NodeOut elem, const GraphDefBuilder::Options&
                opts) {
  static const string kOpName = "StackPush";
  return BinaryOp(kOpName, handle, elem, opts);
}

Node* TensorArray(NodeOut size, DataType dtype, const GraphDefBuilder::Options&
                  opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "TensorArray";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(size);
  node_builder.Attr("dtype", dtype);
  return opts.FinalizeBuilder(&node_builder);
}

Node* TensorArrayClose(NodeOut handle, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "TensorArrayClose";
  return UnaryOp(kOpName, handle, opts);
}

Node* TensorArrayGrad(NodeOut handle, StringPiece source, const
                      GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "TensorArrayGrad";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(handle);
  node_builder.Attr("source", source);
  return opts.FinalizeBuilder(&node_builder);
}

Node* TensorArrayPack(NodeOut handle, NodeOut flow_in, DataType dtype, const
                      GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "TensorArrayPack";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(handle);
  node_builder.Input(flow_in);
  node_builder.Attr("dtype", dtype);
  return opts.FinalizeBuilder(&node_builder);
}

Node* TensorArrayRead(NodeOut handle, NodeOut index, NodeOut flow_in, DataType
                      dtype, const GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "TensorArrayRead";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(handle);
  node_builder.Input(index);
  node_builder.Input(flow_in);
  node_builder.Attr("dtype", dtype);
  return opts.FinalizeBuilder(&node_builder);
}

Node* TensorArrayUnpack(NodeOut handle, NodeOut value, NodeOut flow_in, const
                        GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "TensorArrayUnpack";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(handle);
  node_builder.Input(value);
  node_builder.Input(flow_in);
  return opts.FinalizeBuilder(&node_builder);
}

Node* TensorArrayWrite(NodeOut handle, NodeOut index, NodeOut value, NodeOut
                       flow_in, const GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "TensorArrayWrite";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(handle);
  node_builder.Input(index);
  node_builder.Input(value);
  node_builder.Input(flow_in);
  return opts.FinalizeBuilder(&node_builder);
}

}  // namespace ops
}  // namespace tensorflow
