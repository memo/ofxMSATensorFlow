// This file is MACHINE GENERATED! Do not edit.

#ifndef TENSORFLOW_CC_OPS_IO_OPS_H_
#define TENSORFLOW_CC_OPS_IO_OPS_H_

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


// A Reader that outputs fixed-length records from a file.
//
// Arguments:
// * opts:
//   .WithAttr("header_bytes", int64): Defaults to 0.
//   .WithAttr("footer_bytes", int64): Defaults to 0.
//   .WithAttr("container", StringPiece): Defaults to "".
//     If non-empty, this reader is placed in the given container.
// Otherwise, a default container is used.
//   .WithAttr("shared_name", StringPiece): Defaults to "".
//     If non-empty, this reader is named in the given bucket
// with this shared_name. Otherwise, the node name is used instead.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The handle to reference the Reader.
Node* FixedLengthRecordReader(int64 record_bytes, const
                              GraphDefBuilder::Options& opts);

// A Reader that outputs the queued work as both the key and value.
//
// To use, enqueue strings in a Queue.  ReaderRead will take the front
// work string and output (work, work).
//
// Arguments:
// * opts:
//   .WithAttr("container", StringPiece): Defaults to "".
//     If non-empty, this reader is placed in the given container.
// Otherwise, a default container is used.
//   .WithAttr("shared_name", StringPiece): Defaults to "".
//     If non-empty, this reader is named in the given bucket
// with this shared_name. Otherwise, the node name is used instead.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The handle to reference the Reader.
Node* IdentityReader(const GraphDefBuilder::Options& opts);

// Returns the set of files matching a pattern.
//
// Note that this routine only supports wildcard characters in the
// basename portion of the pattern, not in the directory portion.
//
// Arguments:
// * pattern: A (scalar) shell wildcard pattern.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// A vector of matching filenames.
Node* MatchingFiles(NodeOut pattern, const GraphDefBuilder::Options& opts);

// Reads and outputs the entire contents of the input filename.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* ReadFile(NodeOut filename, const GraphDefBuilder::Options& opts);

// Returns the number of records this Reader has produced.
//
// This is the same as the number of ReaderRead executions that have
// succeeded.
//
// Arguments:
// * reader_handle: Handle to a Reader.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* ReaderNumRecordsProduced(NodeOut reader_handle, const
                               GraphDefBuilder::Options& opts);

// Returns the number of work units this Reader has finished processing.
//
// Arguments:
// * reader_handle: Handle to a Reader.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* ReaderNumWorkUnitsCompleted(NodeOut reader_handle, const
                                  GraphDefBuilder::Options& opts);

// Returns the next record (key, value pair) produced by a Reader.
//
// Will dequeue from the input queue if necessary (e.g. when the
// Reader needs to start reading from a new file since it has finished
// with the previous file).
//
// Arguments:
// * reader_handle: Handle to a Reader.
// * queue_handle: Handle to a Queue, with string work items.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with outputs:
// * key: A scalar.
// * value: A scalar.
Node* ReaderRead(NodeOut reader_handle, NodeOut queue_handle, const
                 GraphDefBuilder::Options& opts);

// Restore a Reader to its initial clean state.
//
// Arguments:
// * reader_handle: Handle to a Reader.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* ReaderReset(NodeOut reader_handle, const GraphDefBuilder::Options& opts);

// Restore a reader to a previously saved state.
//
// Not all Readers support being restored, so this can produce an
// Unimplemented error.
//
// Arguments:
// * reader_handle: Handle to a Reader.
// * state: Result of a ReaderSerializeState of a Reader with type
// matching reader_handle.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* ReaderRestoreState(NodeOut reader_handle, NodeOut state, const
                         GraphDefBuilder::Options& opts);

// Produce a string tensor that encodes the state of a Reader.
//
// Not all Readers support being serialized, so this can produce an
// Unimplemented error.
//
// Arguments:
// * reader_handle: Handle to a Reader.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* ReaderSerializeState(NodeOut reader_handle, const
                           GraphDefBuilder::Options& opts);

// Restores a tensor from checkpoint files.
//
// Reads a tensor stored in one or several files. If there are several files (for
// instance because a tensor was saved as slices), `file_pattern` may contain
// wildcard symbols (`*` and `?`) in the filename portion only, not in the
// directory portion.
//
// If a `file_pattern` matches several files, `preferred_shard` can be used to hint
// in which file the requested tensor is likely to be found. This op will first
// open the file at index `preferred_shard` in the list of matching files and try
// to restore tensors from that file.  Only if some tensors or tensor slices are
// not found in that first file, then the Op opens all the files. Setting
// `preferred_shard` to match the value passed as the `shard` input
// of a matching `Save` Op may speed up Restore.  This attribute only affects
// performance, not correctness.  The default value -1 means files are processed in
// order.
//
// See also `RestoreSlice`.
//
// Arguments:
// * file_pattern: Must have a single element. The pattern of the files from
// which we read the tensor.
// * tensor_name: Must have a single element. The name of the tensor to be
// restored.
// * dt: The type of the tensor to be restored.
// * opts:
//   .WithAttr("preferred_shard", int64): Defaults to -1.
//     Index of file to open first if multiple files match
// `file_pattern`.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The restored tensor.
Node* Restore(NodeOut file_pattern, NodeOut tensor_name, DataType dt, const
              GraphDefBuilder::Options& opts);

// Restores a tensor from checkpoint files.
//
// This is like `Restore` except that restored tensor can be listed as filling
// only a slice of a larger tensor.  `shape_and_slice` specifies the shape of the
// larger tensor and the slice that the restored tensor covers.
//
// The `shape_and_slice` input has the same format as the
// elements of the `shapes_and_slices` input of the `SaveSlices` op.
//
// Arguments:
// * file_pattern: Must have a single element. The pattern of the files from
// which we read the tensor.
// * tensor_name: Must have a single element. The name of the tensor to be
// restored.
// * shape_and_slice: Scalar. The shapes and slice specifications to use when
// restoring a tensors.
// * dt: The type of the tensor to be restored.
// * opts:
//   .WithAttr("preferred_shard", int64): Defaults to -1.
//     Index of file to open first if multiple files match
// `file_pattern`. See the documentation for `Restore`.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The restored tensor.
Node* RestoreSlice(NodeOut file_pattern, NodeOut tensor_name, NodeOut
                   shape_and_slice, DataType dt, const
                   GraphDefBuilder::Options& opts);

// Saves the input tensors to disk.
//
// The size of `tensor_names` must match the number of tensors in `data`. `data[i]`
// is written to `filename` with name `tensor_names[i]`.
//
// See also `SaveSlices`.
//
// Arguments:
// * filename: Must have a single element. The name of the file to which we write
// the tensor.
// * tensor_names: Shape `[N]`. The names of the tensors to be saved.
// * data: `N` tensors to save.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* Save(NodeOut filename, NodeOut tensor_names, gtl::ArraySlice<NodeOut>
           data, const GraphDefBuilder::Options& opts);

// Saves input tensors slices to disk.
//
// This is like `Save` except that tensors can be listed in the saved file as being
// a slice of a larger tensor.  `shapes_and_slices` specifies the shape of the
// larger tensor and the slice that this tensor covers. `shapes_and_slices` must
// have as many elements as `tensor_names`.
//
// Elements of the `shapes_and_slices` input must either be:
//
// *  The empty string, in which case the corresponding tensor is
//    saved normally.
// *  A string of the form `dim0 dim1 ... dimN-1 slice-spec` where the
//    `dimI` are the dimensions of the larger tensor and `slice-spec`
//    specifies what part is covered by the tensor to save.
//
// `slice-spec` itself is a `:`-separated list: `slice0:slice1:...:sliceN-1`
// where each `sliceI` is either:
//
// *  The string `-` meaning that the slice covers all indices of this dimension
// *  `start,length` where `start` and `length` are integers.  In that
//    case the slice covers `length` indices starting at `start`.
//
// See also `Save`.
//
// Arguments:
// * filename: Must have a single element. The name of the file to which we write the
// tensor.
// * tensor_names: Shape `[N]`. The names of the tensors to be saved.
// * shapes_and_slices: Shape `[N]`.  The shapes and slice specifications to use when
// saving the tensors.
// * data: `N` tensors to save.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* SaveSlices(NodeOut filename, NodeOut tensor_names, NodeOut
                 shapes_and_slices, gtl::ArraySlice<NodeOut> data, const
                 GraphDefBuilder::Options& opts);

// Generate a sharded filename. The filename is printf formatted as
//
//    %s-%05d-of-%05d, basename, shard, num_shards.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* ShardedFilename(NodeOut basename, NodeOut shard, NodeOut num_shards,
                      const GraphDefBuilder::Options& opts);

// Generate a glob pattern matching all sharded file names.
//
// Arguments:
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* ShardedFilespec(NodeOut basename, NodeOut num_shards, const
                      GraphDefBuilder::Options& opts);

// A Reader that outputs the records from a TensorFlow Records file.
//
// Arguments:
// * opts:
//   .WithAttr("container", StringPiece): Defaults to "".
//     If non-empty, this reader is placed in the given container.
// Otherwise, a default container is used.
//   .WithAttr("shared_name", StringPiece): Defaults to "".
//     If non-empty, this reader is named in the given bucket
// with this shared_name. Otherwise, the node name is used instead.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The handle to reference the Reader.
Node* TFRecordReader(const GraphDefBuilder::Options& opts);

// A Reader that outputs the lines of a file delimited by '\n'.
//
// Arguments:
// * opts:
//   .WithAttr("skip_header_lines", int64): Defaults to 0.
//     Number of lines to skip from the beginning of every file.
//   .WithAttr("container", StringPiece): Defaults to "".
//     If non-empty, this reader is placed in the given container.
// Otherwise, a default container is used.
//   .WithAttr("shared_name", StringPiece): Defaults to "".
//     If non-empty, this reader is named in the given bucket
// with this shared_name. Otherwise, the node name is used instead.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The handle to reference the Reader.
Node* TextLineReader(const GraphDefBuilder::Options& opts);

// A Reader that outputs the entire contents of a file as a value.
//
// To use, enqueue filenames in a Queue.  The output of ReaderRead will
// be a filename (key) and the contents of that file (value).
//
// Arguments:
// * opts:
//   .WithAttr("container", StringPiece): Defaults to "".
//     If non-empty, this reader is placed in the given container.
// Otherwise, a default container is used.
//   .WithAttr("shared_name", StringPiece): Defaults to "".
//     If non-empty, this reader is named in the given bucket
// with this shared_name. Otherwise, the node name is used instead.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The handle to reference the Reader.
Node* WholeFileReader(const GraphDefBuilder::Options& opts);

}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_OPS_IO_OPS_H_
