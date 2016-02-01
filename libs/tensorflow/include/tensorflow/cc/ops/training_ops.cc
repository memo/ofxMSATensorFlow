// This file is MACHINE GENERATED! Do not edit.

#include "tensorflow/cc/ops/training_ops.h"

#include "tensorflow/core/graph/node_builder.h"

namespace tensorflow {
namespace ops {

Node* ApplyAdagrad(NodeOut var, NodeOut accum, NodeOut lr, NodeOut grad, const
                   GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "ApplyAdagrad";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(var);
  node_builder.Input(accum);
  node_builder.Input(lr);
  node_builder.Input(grad);
  return opts.FinalizeBuilder(&node_builder);
}

Node* ApplyAdam(NodeOut var, NodeOut m, NodeOut v, NodeOut beta1_power, NodeOut
                beta2_power, NodeOut lr, NodeOut beta1, NodeOut beta2, NodeOut
                epsilon, NodeOut grad, const GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "ApplyAdam";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(var);
  node_builder.Input(m);
  node_builder.Input(v);
  node_builder.Input(beta1_power);
  node_builder.Input(beta2_power);
  node_builder.Input(lr);
  node_builder.Input(beta1);
  node_builder.Input(beta2);
  node_builder.Input(epsilon);
  node_builder.Input(grad);
  return opts.FinalizeBuilder(&node_builder);
}

Node* ApplyGradientDescent(NodeOut var, NodeOut alpha, NodeOut delta, const
                           GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "ApplyGradientDescent";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(var);
  node_builder.Input(alpha);
  node_builder.Input(delta);
  return opts.FinalizeBuilder(&node_builder);
}

Node* ApplyMomentum(NodeOut var, NodeOut accum, NodeOut lr, NodeOut grad,
                    NodeOut momentum, const GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "ApplyMomentum";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(var);
  node_builder.Input(accum);
  node_builder.Input(lr);
  node_builder.Input(grad);
  node_builder.Input(momentum);
  return opts.FinalizeBuilder(&node_builder);
}

Node* ApplyRMSProp(NodeOut var, NodeOut ms, NodeOut mom, NodeOut lr, NodeOut
                   rho, NodeOut momentum, NodeOut epsilon, NodeOut grad, const
                   GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "ApplyRMSProp";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(var);
  node_builder.Input(ms);
  node_builder.Input(mom);
  node_builder.Input(lr);
  node_builder.Input(rho);
  node_builder.Input(momentum);
  node_builder.Input(epsilon);
  node_builder.Input(grad);
  return opts.FinalizeBuilder(&node_builder);
}

Node* SparseApplyAdagrad(NodeOut var, NodeOut accum, NodeOut lr, NodeOut grad,
                         NodeOut indices, const GraphDefBuilder::Options& opts)
                         {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "SparseApplyAdagrad";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(var);
  node_builder.Input(accum);
  node_builder.Input(lr);
  node_builder.Input(grad);
  node_builder.Input(indices);
  return opts.FinalizeBuilder(&node_builder);
}

Node* SparseApplyMomentum(NodeOut var, NodeOut accum, NodeOut lr, NodeOut grad,
                          NodeOut indices, NodeOut momentum, const
                          GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "SparseApplyMomentum";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(var);
  node_builder.Input(accum);
  node_builder.Input(lr);
  node_builder.Input(grad);
  node_builder.Input(indices);
  node_builder.Input(momentum);
  return opts.FinalizeBuilder(&node_builder);
}

}  // namespace ops
}  // namespace tensorflow
