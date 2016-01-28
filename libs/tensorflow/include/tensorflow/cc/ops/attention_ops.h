// This file is MACHINE GENERATED! Do not edit.

#ifndef TENSORFLOW_CC_OPS_ATTENTION_OPS_H_
#define TENSORFLOW_CC_OPS_ATTENTION_OPS_H_

#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/public/tensor.h"
#include "tensorflow/core/public/tensor_shape.h"

namespace tensorflow {
namespace ops {

// These add a node to the graph from opts.
//
// Note for "NodeOut" inputs, you will typically either pass
// * a {Node*, int index} (to pass the index-th output of that node), or
// * a Node* (to pass the first output of that node).


// Extracts a glimpse from the input tensor.
//
// Returns a set of windows called glimpses extracted at location `offsets`
// from the input tensor. If the windows only partially overlaps the inputs, the
// non overlapping areas will be filled with random noise.
//
// The result is a 4-D tensor of shape `[batch_size, glimpse_height,
// glimpse_width, channels]`. The channels and batch dimensions are the same as that
// of the input tensor. The height and width of the output windows are
// specified in the `size` parameter.
//
// The argument `normalized` and `centered` controls how the windows are built:
// * If the coordinates are normalized but not centered, 0.0 and 1.0
//   correspond to the minimum and maximum of each height and width dimension.
// * If the coordinates are both normalized and centered, they range from -1.0 to
//   1.0. The coordinates (-1.0, -1.0) correspond to the upper left corner, the
//   lower right corner is located at  (1.0, 1.0) and the center is at (0, 0).
// * If the coordinates are not normalized they are interpreted as numbers of pixels.
//
// Arguments:
// * input: A 4-D float tensor of shape `[batch_size, height, width, channels]`.
// * size: A 1-D tensor of 2 elements containing the size of the glimpses to extract.
// The glimpse height must be specified first, following by the glimpse width.
// * offsets: A 2-D integer tensor of shape `[batch_size, 2]` containing the x, y
// locations of the center of each window.
// * opts:
//   .WithAttr("centered", bool): Defaults to true.
//     indicates if the offset coordinates are centered relative to
// the image, in which case the (0, 0) offset is relative to the center of the
// input images. If false, the (0,0) offset corresponds to the upper left corner
// of the input images.
//   .WithAttr("normalized", bool): Defaults to true.
//     indicates if the offset coordinates are normalized.
//   .WithAttr("uniform_noise", bool): Defaults to true.
//     indicates if the noise should be generated using a
// uniform distribution or a gaussian distribution.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// A tensor representing the glimpses `[batch_size, glimpse_height,
// glimpse_width, channels]`.
Node* ExtractGlimpse(NodeOut input, NodeOut size, NodeOut offsets, const
                     GraphDefBuilder::Options& opts);

}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_OPS_ATTENTION_OPS_H_
