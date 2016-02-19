// This file is MACHINE GENERATED! Do not edit.

#include "tensorflow/cc/ops/image_ops.h"

#include "tensorflow/core/graph/node_builder.h"

namespace tensorflow {
namespace ops {

Node* AdjustContrast(NodeOut images, NodeOut contrast_factor, NodeOut
                     min_value, NodeOut max_value, const
                     GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "AdjustContrast";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(images);
  node_builder.Input(contrast_factor);
  node_builder.Input(min_value);
  node_builder.Input(max_value);
  return opts.FinalizeBuilder(&node_builder);
}

Node* AdjustContrastv2(NodeOut images, NodeOut contrast_factor, const
                       GraphDefBuilder::Options& opts) {
  static const string kOpName = "AdjustContrastv2";
  return BinaryOp(kOpName, images, contrast_factor, opts);
}

Node* DecodeJpeg(NodeOut contents, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "DecodeJpeg";
  return UnaryOp(kOpName, contents, opts);
}

Node* DecodePng(NodeOut contents, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "DecodePng";
  return UnaryOp(kOpName, contents, opts);
}

Node* DrawBoundingBoxes(NodeOut images, NodeOut boxes, const
                        GraphDefBuilder::Options& opts) {
  static const string kOpName = "DrawBoundingBoxes";
  return BinaryOp(kOpName, images, boxes, opts);
}

Node* EncodeJpeg(NodeOut image, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "EncodeJpeg";
  return UnaryOp(kOpName, image, opts);
}

Node* EncodePng(NodeOut image, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "EncodePng";
  return UnaryOp(kOpName, image, opts);
}

Node* HSVToRGB(NodeOut images, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "HSVToRGB";
  return UnaryOp(kOpName, images, opts);
}

Node* RGBToHSV(NodeOut images, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "RGBToHSV";
  return UnaryOp(kOpName, images, opts);
}

Node* RandomCrop(NodeOut image, NodeOut size, const GraphDefBuilder::Options&
                 opts) {
  static const string kOpName = "RandomCrop";
  return BinaryOp(kOpName, image, size, opts);
}

Node* ResizeArea(NodeOut images, NodeOut size, const GraphDefBuilder::Options&
                 opts) {
  static const string kOpName = "ResizeArea";
  return BinaryOp(kOpName, images, size, opts);
}

Node* ResizeBicubic(NodeOut images, NodeOut size, const
                    GraphDefBuilder::Options& opts) {
  static const string kOpName = "ResizeBicubic";
  return BinaryOp(kOpName, images, size, opts);
}

Node* ResizeBilinear(NodeOut images, NodeOut size, const
                     GraphDefBuilder::Options& opts) {
  static const string kOpName = "ResizeBilinear";
  return BinaryOp(kOpName, images, size, opts);
}

Node* ResizeBilinearGrad(NodeOut grads, NodeOut original_image, const
                         GraphDefBuilder::Options& opts) {
  static const string kOpName = "ResizeBilinearGrad";
  return BinaryOp(kOpName, grads, original_image, opts);
}

Node* ResizeNearestNeighbor(NodeOut images, NodeOut size, const
                            GraphDefBuilder::Options& opts) {
  static const string kOpName = "ResizeNearestNeighbor";
  return BinaryOp(kOpName, images, size, opts);
}

Node* ResizeNearestNeighborGrad(NodeOut grads, NodeOut size, const
                                GraphDefBuilder::Options& opts) {
  static const string kOpName = "ResizeNearestNeighborGrad";
  return BinaryOp(kOpName, grads, size, opts);
}

Node* SampleDistortedBoundingBox(NodeOut image_size, NodeOut bounding_boxes,
                                 const GraphDefBuilder::Options& opts) {
  static const string kOpName = "SampleDistortedBoundingBox";
  return BinaryOp(kOpName, image_size, bounding_boxes, opts);
}

}  // namespace ops
}  // namespace tensorflow
