// This file is MACHINE GENERATED! Do not edit.

#include "tensorflow/cc/ops/candidate_sampling_ops.h"

#include "tensorflow/core/graph/node_builder.h"

namespace tensorflow {
namespace ops {

Node* AllCandidateSampler(NodeOut true_classes, int64 num_true, int64
                          num_sampled, bool unique, const
                          GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "AllCandidateSampler";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(true_classes);
  node_builder.Attr("num_true", num_true);
  node_builder.Attr("num_sampled", num_sampled);
  node_builder.Attr("unique", unique);
  return opts.FinalizeBuilder(&node_builder);
}

Node* ComputeAccidentalHits(NodeOut true_classes, NodeOut sampled_candidates,
                            int64 num_true, const GraphDefBuilder::Options&
                            opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "ComputeAccidentalHits";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(true_classes);
  node_builder.Input(sampled_candidates);
  node_builder.Attr("num_true", num_true);
  return opts.FinalizeBuilder(&node_builder);
}

Node* FixedUnigramCandidateSampler(NodeOut true_classes, int64 num_true, int64
                                   num_sampled, bool unique, int64 range_max,
                                   const GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "FixedUnigramCandidateSampler";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(true_classes);
  node_builder.Attr("num_true", num_true);
  node_builder.Attr("num_sampled", num_sampled);
  node_builder.Attr("unique", unique);
  node_builder.Attr("range_max", range_max);
  return opts.FinalizeBuilder(&node_builder);
}

Node* LearnedUnigramCandidateSampler(NodeOut true_classes, int64 num_true,
                                     int64 num_sampled, bool unique, int64
                                     range_max, const GraphDefBuilder::Options&
                                     opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "LearnedUnigramCandidateSampler";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(true_classes);
  node_builder.Attr("num_true", num_true);
  node_builder.Attr("num_sampled", num_sampled);
  node_builder.Attr("unique", unique);
  node_builder.Attr("range_max", range_max);
  return opts.FinalizeBuilder(&node_builder);
}

Node* LogUniformCandidateSampler(NodeOut true_classes, int64 num_true, int64
                                 num_sampled, bool unique, int64 range_max,
                                 const GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "LogUniformCandidateSampler";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(true_classes);
  node_builder.Attr("num_true", num_true);
  node_builder.Attr("num_sampled", num_sampled);
  node_builder.Attr("unique", unique);
  node_builder.Attr("range_max", range_max);
  return opts.FinalizeBuilder(&node_builder);
}

Node* ThreadUnsafeUnigramCandidateSampler(NodeOut true_classes, int64 num_true,
                                          int64 num_sampled, bool unique, int64
                                          range_max, const
                                          GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "ThreadUnsafeUnigramCandidateSampler";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(true_classes);
  node_builder.Attr("num_true", num_true);
  node_builder.Attr("num_sampled", num_sampled);
  node_builder.Attr("unique", unique);
  node_builder.Attr("range_max", range_max);
  return opts.FinalizeBuilder(&node_builder);
}

Node* UniformCandidateSampler(NodeOut true_classes, int64 num_true, int64
                              num_sampled, bool unique, int64 range_max, const
                              GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "UniformCandidateSampler";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(true_classes);
  node_builder.Attr("num_true", num_true);
  node_builder.Attr("num_sampled", num_sampled);
  node_builder.Attr("unique", unique);
  node_builder.Attr("range_max", range_max);
  return opts.FinalizeBuilder(&node_builder);
}

}  // namespace ops
}  // namespace tensorflow
