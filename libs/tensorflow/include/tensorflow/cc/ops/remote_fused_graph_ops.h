// This file is MACHINE GENERATED! Do not edit.

#ifndef TENSORFLOW_CC_OPS_REMOTE_FUSED_GRAPH_OPS_H_
#define TENSORFLOW_CC_OPS_REMOTE_FUSED_GRAPH_OPS_H_

// This file is MACHINE GENERATED! Do not edit.

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {
namespace ops {

/// @defgroup remote_fused_graph_ops Remote Fused Graph Ops
/// @{

/// Execute a sub graph on a remote processor transferred by GraphTransferer.
///
/// The graph specifications are serialized by protobuf as graph_transfer_info.
/// The implementation / limitations may differ for each platform
/// and each available peripheral.
///
/// Arguments:
/// * scope: A Scope object
///
/// Returns:
/// * `OutputList`: The output tensor.
class RemoteFusedGraphExecute {
 public:
  RemoteFusedGraphExecute(const ::tensorflow::Scope& scope,
                        ::tensorflow::InputList values, int64 N, StringPiece
                        serialized_graph_transfer_info);
  ::tensorflow::Output operator[](size_t index) const { return output[index]; }


  ::tensorflow::OutputList output;
};

/// @}

}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_OPS_REMOTE_FUSED_GRAPH_OPS_H_
