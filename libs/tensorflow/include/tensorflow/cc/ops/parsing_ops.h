// This file is MACHINE GENERATED! Do not edit.

#ifndef TENSORFLOW_CC_OPS_PARSING_OPS_H_
#define TENSORFLOW_CC_OPS_PARSING_OPS_H_

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


// Convert CSV records to tensors. Each column maps to one tensor.
//
// RFC 4180 format is expected for the CSV records.
// (https://tools.ietf.org/html/rfc4180)
// Note that we allow leading and trailing spaces with int or float field.
//
// Arguments:
// * records: Each string is a record/row in the csv and all records should have
// the same format.
// * record_defaults: One tensor per column of the input record, with either a
// scalar default value for that column or empty if the column is required.
// * opts:
//   .WithAttr("field_delim", StringPiece): Defaults to ",".
//     delimiter to separate fields in a record.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Each tensor will have the same shape as records.
Node* DecodeCSV(NodeOut records, gtl::ArraySlice<NodeOut> record_defaults,
                const GraphDefBuilder::Options& opts);

// Convert JSON-encoded Example records to binary protocol buffer strings.
//
// This op translates a tensor containing Example records, encoded using
// the [standard JSON
// mapping](https://developers.google.com/protocol-buffers/docs/proto3#json),
// into a tensor containing the same records encoded as binary protocol
// buffers. The resulting tensor can then be fed to any of the other
// Example-parsing ops.
//
// Arguments:
// * json_examples: Each string is a JSON object serialized according to the JSON
// mapping of the Example proto.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Each string is a binary Example protocol buffer corresponding
// to the respective element of `json_examples`.
Node* DecodeJSONExample(NodeOut json_examples, const GraphDefBuilder::Options&
                        opts);

// Reinterpret the bytes of a string as a vector of numbers.
//
// Arguments:
// * bytes: All the elements must have the same length.
// * opts:
//   .WithAttr("little_endian", bool): Defaults to true.
//     Whether the input `bytes` are in little-endian order.
// Ignored for `out_type` values that are stored in a single byte like
// `uint8`.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// A Tensor with one more dimension than the input `bytes`.  The
// added dimension will have size equal to the length of the elements
// of `bytes` divided by the number of bytes to represent `out_type`.
Node* DecodeRaw(NodeOut bytes, DataType out_type, const
                GraphDefBuilder::Options& opts);

// Transforms a vector of brain.Example protos (as strings) into typed tensors.
//
// Arguments:
// * serialized: A vector containing a batch of binary serialized Example protos.
// * names: A vector containing the names of the serialized protos.
// May contain, for example, table key (descriptive) names for the
// corresponding serialized protos.  These are purely useful for debugging
// purposes, and the presence of values here has no effect on the output.
// May also be an empty vector if no names are available.
// If non-empty, this vector must be the same length as "serialized".
// * sparse_keys: A list of Nsparse string Tensors (scalars).
// The keys expected in the Examples' features associated with sparse values.
// * dense_keys: A list of Ndense string Tensors (scalars).
// The keys expected in the Examples' features associated with dense values.
// * dense_defaults: A list of Ndense Tensors (some may be empty).
// dense_defaults[j] provides default values
// when the example's feature_map lacks dense_key[j].  If an empty Tensor is
// provided for dense_defaults[j], then the Feature dense_keys[j] is required.
// The input type is inferred from dense_defaults[j], even when it's empty.
// If dense_defaults[j] is not empty, its shape must match dense_shapes[j].
// * sparse_types: A list of Nsparse types; the data types of data in each Feature
// given in sparse_keys.
// Currently the ParseExample supports DT_FLOAT (FloatList),
// DT_INT64 (Int64List), and DT_STRING (BytesList).
// * dense_shapes: A list of Ndense shapes; the shapes of data in each Feature
// given in dense_keys.
// The number of elements in the Feature corresponding to dense_key[j]
// must always equal dense_shapes[j].NumEntries().
// If dense_shapes[j] == (D0, D1, ..., DN) then the shape of output
// Tensor dense_values[j] will be (|serialized|, D0, D1, ..., DN):
// The dense outputs are just the inputs row-stacked by batch.
// * opts:
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with outputs:
// * sparse_indices
// * sparse_values
// * sparse_shapes
// * dense_values
Node* ParseExample(NodeOut serialized, NodeOut names, gtl::ArraySlice<NodeOut>
                   sparse_keys, gtl::ArraySlice<NodeOut> dense_keys,
                   gtl::ArraySlice<NodeOut> dense_defaults, DataTypeSlice
                   sparse_types, gtl::ArraySlice<TensorShape> dense_shapes,
                   const GraphDefBuilder::Options& opts);

// Transforms a scalar brain.SequenceExample proto (as strings) into typed tensors.
//
// Arguments:
// * serialized: A scalar containing a binary serialized SequenceExample proto.
// * feature_list_dense_missing_assumed_empty: A vector listing the
// FeatureList keys which may be missing from the SequenceExample.  If the
// associated FeatureList is missing, it is treated as empty.  By default,
// any FeatureList not listed in this vector must exist in the SequenceExample.
// * context_sparse_keys: A list of Ncontext_sparse string Tensors (scalars).
// The keys expected in the Examples' features associated with context_sparse
// values.
// * context_dense_keys: A list of Ncontext_dense string Tensors (scalars).
// The keys expected in the SequenceExamples' context features associated with
// dense values.
// * feature_list_sparse_keys: A list of Nfeature_list_sparse string Tensors
// (scalars).  The keys expected in the FeatureLists associated with sparse
// values.
// * feature_list_dense_keys: A list of Nfeature_list_dense string Tensors (scalars).
// The keys expected in the SequenceExamples' feature_lists associated
// with lists of dense values.
// * context_dense_defaults: A list of Ncontext_dense Tensors (some may be empty).
// context_dense_defaults[j] provides default values
// when the SequenceExample's context map lacks context_dense_key[j].
// If an empty Tensor is provided for context_dense_defaults[j],
// then the Feature context_dense_keys[j] is required.
// The input type is inferred from context_dense_defaults[j], even when it's
// empty.  If context_dense_defaults[j] is not empty, its shape must match
// context_dense_shapes[j].
// * debug_name: A scalar containing the name of the serialized proto.
// May contain, for example, table key (descriptive) name for the
// corresponding serialized proto.  This is purely useful for debugging
// purposes, and the presence of values here has no effect on the output.
// May also be an empty scalar if no name is available.
// * opts:
//   .WithAttr("context_sparse_types", DataTypeSlice): Defaults to [].
//     A list of Ncontext_sparse types; the data types of data in
// each context Feature given in context_sparse_keys.
// Currently the ParseSingleSequenceExample supports DT_FLOAT (FloatList),
// DT_INT64 (Int64List), and DT_STRING (BytesList).
//   .WithAttr("feature_list_dense_types", DataTypeSlice): Defaults to [].
//   .WithAttr("context_dense_shapes", gtl::ArraySlice<TensorShape>): Defaults to [].
//     A list of Ncontext_dense shapes; the shapes of data in
// each context Feature given in context_dense_keys.
// The number of elements in the Feature corresponding to context_dense_key[j]
// must always equal context_dense_shapes[j].NumEntries().
// The shape of context_dense_values[j] will match context_dense_shapes[j].
//   .WithAttr("feature_list_sparse_types", DataTypeSlice): Defaults to [].
//     A list of Nfeature_list_sparse types; the data types
// of data in each FeatureList given in feature_list_sparse_keys.
// Currently the ParseSingleSequenceExample supports DT_FLOAT (FloatList),
// DT_INT64 (Int64List), and DT_STRING (BytesList).
//   .WithAttr("feature_list_dense_shapes", gtl::ArraySlice<TensorShape>): Defaults to [].
//     A list of Nfeature_list_dense shapes; the shapes of
// data in each FeatureList given in feature_list_dense_keys.
// The shape of each Feature in the FeatureList corresponding to
// feature_list_dense_key[j] must always equal
// feature_list_dense_shapes[j].NumEntries().
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with outputs:
// * context_sparse_indices
// * context_sparse_values
// * context_sparse_shapes
// * context_dense_values
// * feature_list_sparse_indices
// * feature_list_sparse_values
// * feature_list_sparse_shapes
// * feature_list_dense_values
Node* ParseSingleSequenceExample(NodeOut serialized, NodeOut
                                 feature_list_dense_missing_assumed_empty,
                                 gtl::ArraySlice<NodeOut> context_sparse_keys,
                                 gtl::ArraySlice<NodeOut> context_dense_keys,
                                 gtl::ArraySlice<NodeOut>
                                 feature_list_sparse_keys,
                                 gtl::ArraySlice<NodeOut>
                                 feature_list_dense_keys,
                                 gtl::ArraySlice<NodeOut>
                                 context_dense_defaults, NodeOut debug_name,
                                 const GraphDefBuilder::Options& opts);

// Converts each string in the input Tensor to the specified numeric type.
//
// (Note that int32 overflow results in an error while float overflow
// results in a rounded value.)
//
// Arguments:
// * opts:
//   .WithAttr("out_type", DataType): Defaults to DT_FLOAT.
//     The numeric type to interpret each string in string_tensor as.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// A Tensor of the same shape as the input `string_tensor`.
Node* StringToNumber(NodeOut string_tensor, const GraphDefBuilder::Options&
                     opts);

}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_OPS_PARSING_OPS_H_
