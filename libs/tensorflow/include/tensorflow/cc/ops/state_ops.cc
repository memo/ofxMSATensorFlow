// This file is MACHINE GENERATED! Do not edit.

#include "tensorflow/cc/ops/state_ops.h"

#include "tensorflow/core/graph/node_builder.h"

namespace tensorflow {
namespace ops {

Node* Assign(NodeOut ref, NodeOut value, const GraphDefBuilder::Options& opts)
             {
  static const string kOpName = "Assign";
  return BinaryOp(kOpName, ref, value, opts);
}

Node* AssignAdd(NodeOut ref, NodeOut value, const GraphDefBuilder::Options&
                opts) {
  static const string kOpName = "AssignAdd";
  return BinaryOp(kOpName, ref, value, opts);
}

Node* AssignSub(NodeOut ref, NodeOut value, const GraphDefBuilder::Options&
                opts) {
  static const string kOpName = "AssignSub";
  return BinaryOp(kOpName, ref, value, opts);
}

Node* CountUpTo(NodeOut ref, int64 limit, const GraphDefBuilder::Options& opts)
                {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "CountUpTo";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(ref);
  node_builder.Attr("limit", limit);
  return opts.FinalizeBuilder(&node_builder);
}

Node* DestroyTemporaryVariable(NodeOut ref, StringPiece var_name, const
                               GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "DestroyTemporaryVariable";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(ref);
  node_builder.Attr("var_name", var_name);
  return opts.FinalizeBuilder(&node_builder);
}

Node* ScatterAdd(NodeOut ref, NodeOut indices, NodeOut updates, const
                 GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "ScatterAdd";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(ref);
  node_builder.Input(indices);
  node_builder.Input(updates);
  return opts.FinalizeBuilder(&node_builder);
}

Node* ScatterSub(NodeOut ref, NodeOut indices, NodeOut updates, const
                 GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "ScatterSub";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(ref);
  node_builder.Input(indices);
  node_builder.Input(updates);
  return opts.FinalizeBuilder(&node_builder);
}

Node* ScatterUpdate(NodeOut ref, NodeOut indices, NodeOut updates, const
                    GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "ScatterUpdate";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(ref);
  node_builder.Input(indices);
  node_builder.Input(updates);
  return opts.FinalizeBuilder(&node_builder);
}

Node* TemporaryVariable(TensorShape shape, DataType dtype, const
                        GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "TemporaryVariable";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Attr("shape", shape);
  node_builder.Attr("dtype", dtype);
  return opts.FinalizeBuilder(&node_builder);
}

Node* Variable(TensorShape shape, DataType dtype, const
               GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "Variable";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Attr("shape", shape);
  node_builder.Attr("dtype", dtype);
  return opts.FinalizeBuilder(&node_builder);
}

}  // namespace ops
}  // namespace tensorflow
