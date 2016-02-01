// This file is MACHINE GENERATED! Do not edit.

#include "tensorflow/cc/ops/sendrecv_ops.h"

#include "tensorflow/core/graph/node_builder.h"

namespace tensorflow {
namespace ops {

Node* _HostRecv(DataType tensor_type, StringPiece tensor_name, StringPiece
                send_device, int64 send_device_incarnation, StringPiece
                recv_device, const GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "_HostRecv";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Attr("tensor_type", tensor_type);
  node_builder.Attr("tensor_name", tensor_name);
  node_builder.Attr("send_device", send_device);
  node_builder.Attr("send_device_incarnation", send_device_incarnation);
  node_builder.Attr("recv_device", recv_device);
  return opts.FinalizeBuilder(&node_builder);
}

Node* _HostSend(NodeOut tensor, StringPiece tensor_name, StringPiece
                send_device, int64 send_device_incarnation, StringPiece
                recv_device, const GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "_HostSend";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(tensor);
  node_builder.Attr("tensor_name", tensor_name);
  node_builder.Attr("send_device", send_device);
  node_builder.Attr("send_device_incarnation", send_device_incarnation);
  node_builder.Attr("recv_device", recv_device);
  return opts.FinalizeBuilder(&node_builder);
}

Node* _Recv(DataType tensor_type, StringPiece tensor_name, StringPiece
            send_device, int64 send_device_incarnation, StringPiece
            recv_device, const GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "_Recv";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Attr("tensor_type", tensor_type);
  node_builder.Attr("tensor_name", tensor_name);
  node_builder.Attr("send_device", send_device);
  node_builder.Attr("send_device_incarnation", send_device_incarnation);
  node_builder.Attr("recv_device", recv_device);
  return opts.FinalizeBuilder(&node_builder);
}

Node* _Send(NodeOut tensor, StringPiece tensor_name, StringPiece send_device,
            int64 send_device_incarnation, StringPiece recv_device, const
            GraphDefBuilder::Options& opts) {
  if (opts.HaveError()) return nullptr;
  static const string kOpName = "_Send";
  NodeBuilder node_builder(opts.GetNameForOp(kOpName), kOpName,
                           opts.op_registry());
  node_builder.Input(tensor);
  node_builder.Attr("tensor_name", tensor_name);
  node_builder.Attr("send_device", send_device);
  node_builder.Attr("send_device_incarnation", send_device_incarnation);
  node_builder.Attr("recv_device", recv_device);
  return opts.FinalizeBuilder(&node_builder);
}

}  // namespace ops
}  // namespace tensorflow
