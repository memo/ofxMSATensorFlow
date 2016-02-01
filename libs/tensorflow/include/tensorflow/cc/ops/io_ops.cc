// This file is MACHINE GENERATED! Do not edit.

#include "tensorflow/cc/ops/io_ops.h"

#include "tensorflow/core/graph/node_builder.h"

namespace tensorflow {
namespace ops {

Node* FixedLengthRecordReader(int64 record_bytes, const
                              GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "FixedLengthRecordReader";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Attr("record_bytes", record_bytes);
  return opts.FinalizeBuilder(&node_builder);
}

Node* IdentityReader(const GraphDefBuilder::Options& opts) {
  static const string kOpName = "IdentityReader";
  return SourceOp(kOpName, opts);
}

Node* MatchingFiles(NodeOut pattern, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "MatchingFiles";
  return UnaryOp(kOpName, pattern, opts);
}

Node* ReadFile(NodeOut filename, const GraphDefBuilder::Options& opts) {
  static const string kOpName = "ReadFile";
  return UnaryOp(kOpName, filename, opts);
}

Node* ReaderNumRecordsProduced(NodeOut reader_handle, const
                               GraphDefBuilder::Options& opts) {
  static const string kOpName = "ReaderNumRecordsProduced";
  return UnaryOp(kOpName, reader_handle, opts);
}

Node* ReaderNumWorkUnitsCompleted(NodeOut reader_handle, const
                                  GraphDefBuilder::Options& opts) {
  static const string kOpName = "ReaderNumWorkUnitsCompleted";
  return UnaryOp(kOpName, reader_handle, opts);
}

Node* ReaderRead(NodeOut reader_handle, NodeOut queue_handle, const
                 GraphDefBuilder::Options& opts) {
  static const string kOpName = "ReaderRead";
  return BinaryOp(kOpName, reader_handle, queue_handle, opts);
}

Node* ReaderReset(NodeOut reader_handle, const GraphDefBuilder::Options& opts)
                  {
  static const string kOpName = "ReaderReset";
  return UnaryOp(kOpName, reader_handle, opts);
}

Node* ReaderRestoreState(NodeOut reader_handle, NodeOut state, const
                         GraphDefBuilder::Options& opts) {
  static const string kOpName = "ReaderRestoreState";
  return BinaryOp(kOpName, reader_handle, state, opts);
}

Node* ReaderSerializeState(NodeOut reader_handle, const
                           GraphDefBuilder::Options& opts) {
  static const string kOpName = "ReaderSerializeState";
  return UnaryOp(kOpName, reader_handle, opts);
}

Node* Restore(NodeOut file_pattern, NodeOut tensor_name, DataType dt, const
              GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "Restore";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(file_pattern);
  node_builder.Input(tensor_name);
  node_builder.Attr("dt", dt);
  return opts.FinalizeBuilder(&node_builder);
}

Node* RestoreSlice(NodeOut file_pattern, NodeOut tensor_name, NodeOut
                   shape_and_slice, DataType dt, const
                   GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "RestoreSlice";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(file_pattern);
  node_builder.Input(tensor_name);
  node_builder.Input(shape_and_slice);
  node_builder.Attr("dt", dt);
  return opts.FinalizeBuilder(&node_builder);
}

Node* Save(NodeOut filename, NodeOut tensor_names, gtl::ArraySlice<NodeOut>
           data, const GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "Save";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(filename);
  node_builder.Input(tensor_names);
  node_builder.Input(data);
  return opts.FinalizeBuilder(&node_builder);
}

Node* SaveSlices(NodeOut filename, NodeOut tensor_names, NodeOut
                 shapes_and_slices, gtl::ArraySlice<NodeOut> data, const
                 GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "SaveSlices";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(filename);
  node_builder.Input(tensor_names);
  node_builder.Input(shapes_and_slices);
  node_builder.Input(data);
  return opts.FinalizeBuilder(&node_builder);
}

Node* ShardedFilename(NodeOut basename, NodeOut shard, NodeOut num_shards,
                      const GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "ShardedFilename";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(basename);
  node_builder.Input(shard);
  node_builder.Input(num_shards);
  return opts.FinalizeBuilder(&node_builder);
}

Node* ShardedFilespec(NodeOut basename, NodeOut num_shards, const
                      GraphDefBuilder::Options& opts) {
  static const string kOpName = "ShardedFilespec";
  return BinaryOp(kOpName, basename, num_shards, opts);
}

Node* TFRecordReader(const GraphDefBuilder::Options& opts) {
  static const string kOpName = "TFRecordReader";
  return SourceOp(kOpName, opts);
}

Node* TextLineReader(const GraphDefBuilder::Options& opts) {
  static const string kOpName = "TextLineReader";
  return SourceOp(kOpName, opts);
}

Node* WholeFileReader(const GraphDefBuilder::Options& opts) {
  static const string kOpName = "WholeFileReader";
  return SourceOp(kOpName, opts);
}

}  // namespace ops
}  // namespace tensorflow
