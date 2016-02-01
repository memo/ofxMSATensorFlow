// This file is MACHINE GENERATED! Do not edit.

#include "tensorflow/cc/ops/control_flow_ops.h"

#include "tensorflow/core/graph/node_builder.h"

namespace tensorflow {
namespace ops {

Node* ControlTrigger(const GraphDefBuilder::Options& opts) {
  static const string kOpName = "ControlTrigger";
  return SourceOp(kOpName, opts);
}

Node* Enter(NodeOut data, StringPiece frame_name, const
            GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "Enter";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(data);
  node_builder.Attr("frame_name", frame_name);
  return opts.FinalizeBuilder(&node_builder);
}

Node* Exit(NodeOut data, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "Exit";
  return UnaryOp(kOpName, data, opts);
}

Node* LoopCond(NodeOut input, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "LoopCond";
  return UnaryOp(kOpName, input, opts);
}

Node* Merge(gtl::ArraySlice<NodeOut> inputs, const GraphDefBuilder::Options&
            opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "Merge";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(inputs);
  return opts.FinalizeBuilder(&node_builder);
}

Node* NextIteration(NodeOut data, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "NextIteration";
  return UnaryOp(kOpName, data, opts);
}

Node* RefEnter(NodeOut data, StringPiece frame_name, const
               GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "RefEnter";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(data);
  node_builder.Attr("frame_name", frame_name);
  return opts.FinalizeBuilder(&node_builder);
}

Node* RefExit(NodeOut data, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "RefExit";
  return UnaryOp(kOpName, data, opts);
}

Node* RefMerge(gtl::ArraySlice<NodeOut> inputs, const GraphDefBuilder::Options&
               opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "RefMerge";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(inputs);
  return opts.FinalizeBuilder(&node_builder);
}

Node* RefNextIteration(NodeOut data, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "RefNextIteration";
  return UnaryOp(kOpName, data, opts);
}

Node* RefSelect(NodeOut index, gtl::ArraySlice<NodeOut> inputs, const
                GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "RefSelect";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(index);
  node_builder.Input(inputs);
  return opts.FinalizeBuilder(&node_builder);
}

Node* RefSwitch(NodeOut data, NodeOut pred, const GraphDefBuilder::Options&
                opts) {
  static const string kOpName = "RefSwitch";
  return BinaryOp(kOpName, data, pred, opts);
}

Node* Switch(NodeOut data, NodeOut pred, const GraphDefBuilder::Options& opts)
             {
  static const string kOpName = "Switch";
  return BinaryOp(kOpName, data, pred, opts);
}

}  // namespace ops
}  // namespace tensorflow
