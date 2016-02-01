// This file is MACHINE GENERATED! Do not edit.

#include "tensorflow/cc/ops/nn_ops.h"

#include "tensorflow/core/graph/node_builder.h"

namespace tensorflow {
namespace ops {

Node* AvgPool(NodeOut value, gtl::ArraySlice<int> ksize, gtl::ArraySlice<int>
              strides, StringPiece padding, const GraphDefBuilder::Options&
              opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "AvgPool";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(value);
  node_builder.Attr("ksize", ksize);
  node_builder.Attr("strides", strides);
  node_builder.Attr("padding", padding);
  return opts.FinalizeBuilder(&node_builder);
}

Node* AvgPoolGrad(NodeOut orig_input_shape, NodeOut grad, gtl::ArraySlice<int>
                  ksize, gtl::ArraySlice<int> strides, StringPiece padding,
                  const GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "AvgPoolGrad";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(orig_input_shape);
  node_builder.Input(grad);
  node_builder.Attr("ksize", ksize);
  node_builder.Attr("strides", strides);
  node_builder.Attr("padding", padding);
  return opts.FinalizeBuilder(&node_builder);
}

Node* BatchNormWithGlobalNormalization(NodeOut t, NodeOut m, NodeOut v, NodeOut
                                       beta, NodeOut gamma, float
                                       variance_epsilon, bool
                                       scale_after_normalization, const
                                       GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "BatchNormWithGlobalNormalization";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(t);
  node_builder.Input(m);
  node_builder.Input(v);
  node_builder.Input(beta);
  node_builder.Input(gamma);
  node_builder.Attr("variance_epsilon", variance_epsilon);
  node_builder.Attr("scale_after_normalization", scale_after_normalization);
  return opts.FinalizeBuilder(&node_builder);
}

Node* BatchNormWithGlobalNormalizationGrad(NodeOut t, NodeOut m, NodeOut v,
                                           NodeOut gamma, NodeOut backprop,
                                           float variance_epsilon, bool
                                           scale_after_normalization, const
                                           GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "BatchNormWithGlobalNormalizationGrad";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(t);
  node_builder.Input(m);
  node_builder.Input(v);
  node_builder.Input(gamma);
  node_builder.Input(backprop);
  node_builder.Attr("variance_epsilon", variance_epsilon);
  node_builder.Attr("scale_after_normalization", scale_after_normalization);
  return opts.FinalizeBuilder(&node_builder);
}

Node* BiasAdd(NodeOut value, NodeOut bias, const GraphDefBuilder::Options&
              opts) {
  static const string kOpName = "BiasAdd";
  return BinaryOp(kOpName, value, bias, opts);
}

Node* Conv2D(NodeOut input, NodeOut filter, gtl::ArraySlice<int> strides,
             StringPiece padding, const GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "Conv2D";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(input);
  node_builder.Input(filter);
  node_builder.Attr("strides", strides);
  node_builder.Attr("padding", padding);
  return opts.FinalizeBuilder(&node_builder);
}

Node* Conv2DBackpropFilter(NodeOut input, NodeOut filter_sizes, NodeOut
                           out_backprop, gtl::ArraySlice<int> strides,
                           StringPiece padding, const GraphDefBuilder::Options&
                           opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "Conv2DBackpropFilter";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(input);
  node_builder.Input(filter_sizes);
  node_builder.Input(out_backprop);
  node_builder.Attr("strides", strides);
  node_builder.Attr("padding", padding);
  return opts.FinalizeBuilder(&node_builder);
}

Node* Conv2DBackpropInput(NodeOut input_sizes, NodeOut filter, NodeOut
                          out_backprop, gtl::ArraySlice<int> strides,
                          StringPiece padding, const GraphDefBuilder::Options&
                          opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "Conv2DBackpropInput";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(input_sizes);
  node_builder.Input(filter);
  node_builder.Input(out_backprop);
  node_builder.Attr("strides", strides);
  node_builder.Attr("padding", padding);
  return opts.FinalizeBuilder(&node_builder);
}

Node* Elu(NodeOut features, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "Elu";
  return UnaryOp(kOpName, features, opts);
}

Node* EluGrad(NodeOut gradients, NodeOut outputs, const
              GraphDefBuilder::Options& opts) {
  static const string kOpName = "EluGrad";
  return BinaryOp(kOpName, gradients, outputs, opts);
}

Node* InTopK(NodeOut predictions, NodeOut targets, int64 k, const
             GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "InTopK";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(predictions);
  node_builder.Input(targets);
  node_builder.Attr("k", k);
  return opts.FinalizeBuilder(&node_builder);
}

Node* L2Loss(NodeOut t, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "L2Loss";
  return UnaryOp(kOpName, t, opts);
}

Node* LRN(NodeOut input, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "LRN";
  return UnaryOp(kOpName, input, opts);
}

Node* LRNGrad(NodeOut input_grads, NodeOut input_image, NodeOut output_image,
              const GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "LRNGrad";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(input_grads);
  node_builder.Input(input_image);
  node_builder.Input(output_image);
  return opts.FinalizeBuilder(&node_builder);
}

Node* MaxPool(NodeOut input, gtl::ArraySlice<int> ksize, gtl::ArraySlice<int>
              strides, StringPiece padding, const GraphDefBuilder::Options&
              opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "MaxPool";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(input);
  node_builder.Attr("ksize", ksize);
  node_builder.Attr("strides", strides);
  node_builder.Attr("padding", padding);
  return opts.FinalizeBuilder(&node_builder);
}

Node* MaxPoolGrad(NodeOut orig_input, NodeOut orig_output, NodeOut grad,
                  gtl::ArraySlice<int> ksize, gtl::ArraySlice<int> strides,
                  StringPiece padding, const GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "MaxPoolGrad";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(orig_input);
  node_builder.Input(orig_output);
  node_builder.Input(grad);
  node_builder.Attr("ksize", ksize);
  node_builder.Attr("strides", strides);
  node_builder.Attr("padding", padding);
  return opts.FinalizeBuilder(&node_builder);
}

Node* MaxPoolGradWithArgmax(NodeOut input, NodeOut grad, NodeOut argmax,
                            gtl::ArraySlice<int> ksize, gtl::ArraySlice<int>
                            strides, StringPiece padding, const
                            GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "MaxPoolGradWithArgmax";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(input);
  node_builder.Input(grad);
  node_builder.Input(argmax);
  node_builder.Attr("ksize", ksize);
  node_builder.Attr("strides", strides);
  node_builder.Attr("padding", padding);
  return opts.FinalizeBuilder(&node_builder);
}

Node* MaxPoolWithArgmax(NodeOut input, gtl::ArraySlice<int> ksize,
                        gtl::ArraySlice<int> strides, StringPiece padding,
                        const GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "MaxPoolWithArgmax";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(input);
  node_builder.Attr("ksize", ksize);
  node_builder.Attr("strides", strides);
  node_builder.Attr("padding", padding);
  return opts.FinalizeBuilder(&node_builder);
}

Node* Relu(NodeOut features, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "Relu";
  return UnaryOp(kOpName, features, opts);
}

Node* Relu6(NodeOut features, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "Relu6";
  return UnaryOp(kOpName, features, opts);
}

Node* Relu6Grad(NodeOut gradients, NodeOut features, const
                GraphDefBuilder::Options& opts) {
  static const string kOpName = "Relu6Grad";
  return BinaryOp(kOpName, gradients, features, opts);
}

Node* ReluGrad(NodeOut gradients, NodeOut features, const
               GraphDefBuilder::Options& opts) {
  static const string kOpName = "ReluGrad";
  return BinaryOp(kOpName, gradients, features, opts);
}

Node* Softmax(NodeOut logits, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "Softmax";
  return UnaryOp(kOpName, logits, opts);
}

Node* SoftmaxCrossEntropyWithLogits(NodeOut features, NodeOut labels, const
                                    GraphDefBuilder::Options& opts) {
  static const string kOpName = "SoftmaxCrossEntropyWithLogits";
  return BinaryOp(kOpName, features, labels, opts);
}

Node* Softplus(NodeOut features, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "Softplus";
  return UnaryOp(kOpName, features, opts);
}

Node* SoftplusGrad(NodeOut gradients, NodeOut features, const
                   GraphDefBuilder::Options& opts) {
  static const string kOpName = "SoftplusGrad";
  return BinaryOp(kOpName, gradients, features, opts);
}

Node* Softsign(NodeOut features, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "Softsign";
  return UnaryOp(kOpName, features, opts);
}

Node* SoftsignGrad(NodeOut gradients, NodeOut features, const
                   GraphDefBuilder::Options& opts) {
  static const string kOpName = "SoftsignGrad";
  return BinaryOp(kOpName, gradients, features, opts);
}

Node* SparseSoftmaxCrossEntropyWithLogits(NodeOut features, NodeOut labels,
                                          const GraphDefBuilder::Options& opts)
                                          {
  static const string kOpName = "SparseSoftmaxCrossEntropyWithLogits";
  return BinaryOp(kOpName, features, labels, opts);
}

Node* TopK(NodeOut input, int64 k, const GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "TopK";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(input);
  node_builder.Attr("k", k);
  return opts.FinalizeBuilder(&node_builder);
}

Node* TopKV2(NodeOut input, NodeOut k, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "TopKV2";
  return BinaryOp(kOpName, input, k, opts);
}

}  // namespace ops
}  // namespace tensorflow
