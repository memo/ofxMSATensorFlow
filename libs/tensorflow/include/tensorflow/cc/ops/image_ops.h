// This file is MACHINE GENERATED! Do not edit.

#ifndef TENSORFLOW_CC_OPS_IMAGE_OPS_H_
#define TENSORFLOW_CC_OPS_IMAGE_OPS_H_

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {
namespace ops {

// These add a node to the graph from opts.
//
// Note for "NodeOut" inputs, you will typically either pass
// * a {Node*, int index} (to pass the index-th output of that node), or
// * a Node* (to pass the first output of that node).


// Deprecated. Disallowed in GraphDef version >= 2.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* AdjustContrast(NodeOut images, NodeOut contrast_factor, NodeOut
                     min_value, NodeOut max_value, const
                     GraphDefBuilder::Options& opts);

// Adjust the contrast of one or more images.
//
// `images` is a tensor of at least 3 dimensions.  The last 3 dimensions are
// interpreted as `[height, width, channels]`.  The other dimensions only
// represent a collection of images, such as `[batch, height, width, channels].`
//
// Contrast is adjusted independently for each channel of each image.
//
// For each channel, the Op first computes the mean of the image pixels in the
// channel and then adjusts each component of each pixel to
// `(x - mean) * contrast_factor + mean`.
//
// Arguments:
// * images: Images to adjust.  At least 3-D.
// * contrast_factor: A float multiplier for adjusting contrast.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The contrast-adjusted image or images.
Node* AdjustContrastv2(NodeOut images, NodeOut contrast_factor, const
                       GraphDefBuilder::Options& opts);

// Decode a JPEG-encoded image to a uint8 tensor.
//
// The attr `channels` indicates the desired number of color channels for the
// decoded image.
//
// Accepted values are:
//
// *   0: Use the number of channels in the JPEG-encoded image.
// *   1: output a grayscale image.
// *   3: output an RGB image.
//
// If needed, the JPEG-encoded image is transformed to match the requested number
// of color channels.
//
// The attr `ratio` allows downscaling the image by an integer factor during
// decoding.  Allowed values are: 1, 2, 4, and 8.  This is much faster than
// downscaling the image later.
//
// Arguments:
// * contents: 0-D.  The JPEG-encoded image.
// * opts:
//   .WithAttr("channels", int64): Defaults to 0.
//     Number of color channels for the decoded image.
//   .WithAttr("ratio", int64): Defaults to 1.
//     Downscaling ratio.
//   .WithAttr("fancy_upscaling", bool): Defaults to true.
//     If true use a slower but nicer upscaling of the
// chroma planes (yuv420/422 only).
//   .WithAttr("try_recover_truncated", bool): Defaults to false.
//     If true try to recover an image from truncated input.
//   .WithAttr("acceptable_fraction", float): Defaults to 1.
//     The minimum required fraction of lines before a truncated
// input is accepted.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// 3-D with shape `[height, width, channels]`..
Node* DecodeJpeg(NodeOut contents, const GraphDefBuilder::Options& opts);

// Decode a PNG-encoded image to a uint8 or uint16 tensor.
//
// The attr `channels` indicates the desired number of color channels for the
// decoded image.
//
// Accepted values are:
//
// *   0: Use the number of channels in the PNG-encoded image.
// *   1: output a grayscale image.
// *   3: output an RGB image.
// *   4: output an RGBA image.
//
// If needed, the PNG-encoded image is transformed to match the requested number
// of color channels.
//
// Arguments:
// * contents: 0-D.  The PNG-encoded image.
// * opts:
//   .WithAttr("channels", int64): Defaults to 0.
//     Number of color channels for the decoded image.
//   .WithAttr("dtype", DataType): Defaults to DT_UINT8.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// 3-D with shape `[height, width, channels]`.
Node* DecodePng(NodeOut contents, const GraphDefBuilder::Options& opts);

// Draw bounding boxes on a batch of images.
//
// Outputs a copy of `images` but draws on top of the pixels zero or more bounding
// boxes specified by the locations in `boxes`. The coordinates of the each
// bounding box in `boxes are encoded as `[y_min, x_min, y_max, x_max]`. The
// bounding box coordinates are floats in `[0.0, 1.0]` relative to the width and
// height of the underlying image.
//
// For example, if an image is 100 x 200 pixels and the bounding box is
// `[0.1, 0.5, 0.2, 0.9]`, the bottom-left and upper-right coordinates of the
// bounding box will be `(10, 40)` to `(50, 180)`.
//
// Parts of the bounding box may fall outside the image.
//
// Arguments:
// * images: 4-D with shape `[batch, height, width, depth]`. A batch of images.
// * boxes: 3-D with shape `[batch, num_bounding_boxes, 4]` containing bounding
// boxes.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// 4-D with the same shape as `images`. The batch of input images with
// bounding boxes drawn on the images.
Node* DrawBoundingBoxes(NodeOut images, NodeOut boxes, const
                        GraphDefBuilder::Options& opts);

// JPEG-encode an image.
//
// `image` is a 3-D uint8 Tensor of shape `[height, width, channels]`.
//
// The attr `format` can be used to override the color format of the encoded
// output.  Values can be:
//
// *   `''`: Use a default format based on the number of channels in the image.
// *   `grayscale`: Output a grayscale JPEG image.  The `channels` dimension
//     of `image` must be 1.
// *   `rgb`: Output an RGB JPEG image. The `channels` dimension
//     of `image` must be 3.
//
// If `format` is not specified or is the empty string, a default format is picked
// in function of the number of channels in `image`:
//
// *   1: Output a grayscale image.
// *   3: Output an RGB image.
//
// Arguments:
// * image: 3-D with shape `[height, width, channels]`.
// * opts:
//   .WithAttr("format", StringPiece): Defaults to "".
//     Per pixel image format.
//   .WithAttr("quality", int64): Defaults to 95.
//     Quality of the compression from 0 to 100 (higher is better and slower).
//   .WithAttr("progressive", bool): Defaults to false.
//     If True, create a JPEG that loads progressively (coarse to fine).
//   .WithAttr("optimize_size", bool): Defaults to false.
//     If True, spend CPU/RAM to reduce size with no quality change.
//   .WithAttr("chroma_downsampling", bool): Defaults to true.
//     See http://en.wikipedia.org/wiki/Chroma_subsampling.
//   .WithAttr("density_unit", StringPiece): Defaults to "in".
//     Unit used to specify `x_density` and `y_density`:
// pixels per inch (`'in'`) or centimeter (`'cm'`).
//   .WithAttr("x_density", int64): Defaults to 300.
//     Horizontal pixels per density unit.
//   .WithAttr("y_density", int64): Defaults to 300.
//     Vertical pixels per density unit.
//   .WithAttr("xmp_metadata", StringPiece): Defaults to "".
//     If not empty, embed this XMP metadata in the image header.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// 0-D. JPEG-encoded image.
Node* EncodeJpeg(NodeOut image, const GraphDefBuilder::Options& opts);

// PNG-encode an image.
//
// `image` is a 3-D uint8 or uint16 Tensor of shape `[height, width, channels]`
// where `channels` is:
//
// *   1: for grayscale.
// *   2: for grayscale + alpha.
// *   3: for RGB.
// *   4: for RGBA.
//
// The ZLIB compression level, `compression`, can be -1 for the PNG-encoder
// default or a value from 0 to 9.  9 is the highest compression level, generating
// the smallest output, but is slower.
//
// Arguments:
// * image: 3-D with shape `[height, width, channels]`.
// * opts:
//   .WithAttr("compression", int64): Defaults to -1.
//     Compression level.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// 0-D. PNG-encoded image.
Node* EncodePng(NodeOut image, const GraphDefBuilder::Options& opts);

// Convert one or more images from HSV to RGB.
//
// Outputs a tensor of the same shape as the `images` tensor, containing the RGB
// value of the pixels. The output is only well defined if the value in `images`
// are in `[0,1]`.
//
// See `rgb_to_hsv` for a description of the HSV encoding.
//
// Arguments:
// * images: 1-D or higher rank. HSV data to convert. Last dimension must be size 3.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// `images` converted to RGB.
Node* HSVToRGB(NodeOut images, const GraphDefBuilder::Options& opts);

// Converts one or more images from RGB to HSV.
//
// Outputs a tensor of the same shape as the `images` tensor, containing the HSV
// value of the pixels. The output is only well defined if the value in `images`
// are in `[0,1]`.
//
// `output[..., 0]` contains hue, `output[..., 1]` contains saturation, and
// `output[..., 2]` contains value. All HSV values are in `[0,1]`. A hue of 0
// corresponds to pure red, hue 1/3 is pure green, and 2/3 is pure blue.
//
// Arguments:
// * images: 1-D or higher rank. RGB data to convert. Last dimension must be size 3.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// `images` converted to HSV.
Node* RGBToHSV(NodeOut images, const GraphDefBuilder::Options& opts);

// Randomly crop `image`.
//
// `size` is a 1-D int64 tensor with 2 elements representing the crop height and
// width.  The values must be non negative.
//
// This Op picks a random location in `image` and crops a `height` by `width`
// rectangle from that location.  The random location is picked so the cropped
// area will fit inside the original image.
//
// Arguments:
// * image: 3-D of shape `[height, width, channels]`.
// * size: 1-D of length 2 containing: `crop_height`, `crop_width`..
// * opts:
//   .WithAttr("seed", int64): Defaults to 0.
//     If either seed or seed2 are set to be non-zero, the random number
// generator is seeded by the given seed.  Otherwise, it is seeded by a
// random seed.
//   .WithAttr("seed2", int64): Defaults to 0.
//     An second seed to avoid seed collision.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// 3-D of shape `[crop_height, crop_width, channels].`
Node* RandomCrop(NodeOut image, NodeOut size, const GraphDefBuilder::Options&
                 opts);

// Resize `images` to `size` using area interpolation.
//
// Input images can be of different types but output images are always float.
//
// Arguments:
// * images: 4-D with shape `[batch, height, width, channels]`.
// * size: = A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
// new size for the images.
// * opts:
//   .WithAttr("align_corners", bool): Defaults to false.
//     If true, rescale input by (new_height - 1) / (height - 1), which
// exactly aligns the 4 corners of images and resized images. If false, rescale
// by new_height / height. Treat similarly the width dimension.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// 4-D with shape
// `[batch, new_height, new_width, channels]`.
Node* ResizeArea(NodeOut images, NodeOut size, const GraphDefBuilder::Options&
                 opts);

// Resize `images` to `size` using bicubic interpolation.
//
// Input images can be of different types but output images are always float.
//
// Arguments:
// * images: 4-D with shape `[batch, height, width, channels]`.
// * size: = A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
// new size for the images.
// * opts:
//   .WithAttr("align_corners", bool): Defaults to false.
//     If true, rescale input by (new_height - 1) / (height - 1), which
// exactly aligns the 4 corners of images and resized images. If false, rescale
// by new_height / height. Treat similarly the width dimension.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// 4-D with shape
// `[batch, new_height, new_width, channels]`.
Node* ResizeBicubic(NodeOut images, NodeOut size, const
                    GraphDefBuilder::Options& opts);

// Resize `images` to `size` using bilinear interpolation.
//
// Input images can be of different types but output images are always float.
//
// Arguments:
// * images: 4-D with shape `[batch, height, width, channels]`.
// * size: = A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
// new size for the images.
// * opts:
//   .WithAttr("align_corners", bool): Defaults to false.
//     If true, rescale input by (new_height - 1) / (height - 1), which
// exactly aligns the 4 corners of images and resized images. If false, rescale
// by new_height / height. Treat similarly the width dimension.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// 4-D with shape
// `[batch, new_height, new_width, channels]`.
Node* ResizeBilinear(NodeOut images, NodeOut size, const
                     GraphDefBuilder::Options& opts);

// Computes the gradient of bilinear interpolation.
//
// Arguments:
// * grads: 4-D with shape `[batch, height, width, channels]`.
// * original_image: 4-D with shape `[batch, orig_height, orig_width, channels]`,
// The image tensor that was resized.
// * opts:
//   .WithAttr("align_corners", bool): Defaults to false.
//     If true, rescale grads by (orig_height - 1) / (height - 1), which
// exactly aligns the 4 corners of grads and original_image. If false, rescale by
// orig_height / height. Treat similarly the width dimension.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// 4-D with shape `[batch, orig_height, orig_width, channels]`.
// Gradients with respect to the input image. Input image must have been
// float or double.
Node* ResizeBilinearGrad(NodeOut grads, NodeOut original_image, const
                         GraphDefBuilder::Options& opts);

// Resize `images` to `size` using nearest neighbor interpolation.
//
// Arguments:
// * images: 4-D with shape `[batch, height, width, channels]`.
// * size: = A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
// new size for the images.
// * opts:
//   .WithAttr("align_corners", bool): Defaults to false.
//     If true, rescale input by (new_height - 1) / (height - 1), which
// exactly aligns the 4 corners of images and resized images. If false, rescale
// by new_height / height. Treat similarly the width dimension.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// 4-D with shape
// `[batch, new_height, new_width, channels]`.
Node* ResizeNearestNeighbor(NodeOut images, NodeOut size, const
                            GraphDefBuilder::Options& opts);

// Computes the gradient of nearest neighbor interpolation.
//
// Arguments:
// * grads: 4-D with shape `[batch, height, width, channels]`.
// * size: = A 1-D int32 Tensor of 2 elements: `orig_height, orig_width`. The
// original input size.
// * opts:
//   .WithAttr("align_corners", bool): Defaults to false.
//     If true, rescale grads by (orig_height - 1) / (height - 1), which
// exactly aligns the 4 corners of grads and original_image. If false, rescale by
// orig_height / height. Treat similarly the width dimension.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// 4-D with shape `[batch, orig_height, orig_width, channels]`. Gradients
// with respect to the input image.
Node* ResizeNearestNeighborGrad(NodeOut grads, NodeOut size, const
                                GraphDefBuilder::Options& opts);

// Generate a single randomly distorted bounding box for an image.
//
// Bounding box annotations are often supplied in addition to ground-truth labels
// in image recognition or object localization tasks. A common technique for
// training such a system is to randomly distort an image while preserving
// its content, i.e. *data augmentation*. This Op outputs a randomly distorted
// localization of an object, i.e. bounding box, given an `image_size`,
// `bounding_boxes` and a series of constraints.
//
// The output of this Op is a single bounding box that may be used to crop the
// original image. The output is returned as 3 tensors: `begin`, `size` and
// `bboxes`. The first 2 tensors can be fed directly into `tf.slice` to crop the
// image. The latter may be supplied to `tf.image.draw_bounding_box` to visualize
// what the bounding box looks like.
//
// Bounding boxes are supplied and returned as `[y_min, x_min, y_max, x_max]`. The
// bounding box coordinates are floats in `[0.0, 1.0]` relative to the width and
// height of the underlying image.
//
// For example,
//
//     # Generate a single distorted bounding box.
//     begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
//         tf.shape(image),
//         bounding_boxes=bounding_boxes)
//
//     # Draw the bounding box in an image summary.
//     image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0),
//                                                   bbox_for_draw)
//     tf.image_summary('images_with_box', image_with_box)
//
//     # Employ the bounding box to distort the image.
//     distorted_image = tf.slice(image, begin, size)
//
// Note that if no bounding box information is available, setting
// `use_image_if_no_bounding_boxes = true` will assume there is a single implicit
// bounding box covering the whole image. If `use_image_if_no_bounding_boxes` is
// false and no bounding boxes are supplied, an error is raised.
//
// Arguments:
// * image_size: 1-D, containing `[height, width, channels]`.
// * bounding_boxes: 3-D with shape `[batch, N, 4]` describing the N bounding boxes
// associated with the image.
// * opts:
//   .WithAttr("seed", int64): Defaults to 0.
//     If either `seed` or `seed2` are set to non-zero, the random number
// generator is seeded by the given `seed`.  Otherwise, it is seeded by a random
// seed.
//   .WithAttr("seed2", int64): Defaults to 0.
//     A second seed to avoid seed collision.
//   .WithAttr("min_object_covered", float): Defaults to 0.1.
//     The cropped area of the image must contain at least this
// fraction of any bounding box supplied.
//   .WithAttr("aspect_ratio_range", gtl::ArraySlice<float>): Defaults to [0.75, 1.33].
//     The cropped area of the image must have an aspect ratio =
// width / height within this range.
//   .WithAttr("area_range", gtl::ArraySlice<float>): Defaults to [0.05, 1].
//     The cropped area of the image must contain a fraction of the
// supplied image within in this range.
//   .WithAttr("max_attempts", int64): Defaults to 100.
//     Number of attempts at generating a cropped region of the image
// of the specified constraints. After `max_attempts` failures, return the entire
// image.
//   .WithAttr("use_image_if_no_bounding_boxes", bool): Defaults to false.
//     Controls behavior if no bounding boxes supplied.
// If true, assume an implicit bounding box covering the whole input. If false,
// raise an error.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with outputs:
// * begin: 1-D, containing `[offset_height, offset_width, 0]`. Provide as input to
// `tf.slice`.
// * size: 1-D, containing `[target_height, target_width, -1]`. Provide as input to
// `tf.slice`.
// * bboxes: 3-D with shape `[1, 1, 4]` containing the distorted bounding box.
// Provide as input to `tf.image.draw_bounding_boxes`.
Node* SampleDistortedBoundingBox(NodeOut image_size, NodeOut bounding_boxes,
                                 const GraphDefBuilder::Options& opts);

}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_OPS_IMAGE_OPS_H_
