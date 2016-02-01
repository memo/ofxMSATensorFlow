// This file is MACHINE GENERATED! Do not edit.

#ifndef TENSORFLOW_CC_OPS_SENDRECV_OPS_H_
#define TENSORFLOW_CC_OPS_SENDRECV_OPS_H_

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


// Receives the named tensor from send_device on recv_device.
//
// _HostRecv requires its input on host memory whereas _Recv requires its
// input on device memory.
//
// Arguments:
// * tensor_name: The name of the tensor to receive.
// * send_device: The name of the device sending the tensor.
// * send_device_incarnation: The current incarnation of send_device.
// * recv_device: The name of the device receiving the tensor.
// * opts:
//   .WithAttr("client_terminated", bool): Defaults to false.
//     If set to true, this indicates that the node was added
// to the graph as a result of a client-side feed or fetch of Tensor data,
// in which case the corresponding send or recv is expected to be managed
// locally by the caller.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The tensor to receive.
Node* _HostRecv(DataType tensor_type, StringPiece tensor_name, StringPiece
                send_device, int64 send_device_incarnation, StringPiece
                recv_device, const GraphDefBuilder::Options& opts);

// Sends the named tensor from send_device to recv_device.
//
// _HostSend requires its input on host memory whereas _Send requires its
// input on device memory.
//
// Arguments:
// * tensor: The tensor to send.
// * tensor_name: The name of the tensor to send.
// * send_device: The name of the device sending the tensor.
// * send_device_incarnation: The current incarnation of send_device.
// * recv_device: The name of the device receiving the tensor.
// * opts:
//   .WithAttr("client_terminated", bool): Defaults to false.
//     If set to true, this indicates that the node was added
// to the graph as a result of a client-side feed or fetch of Tensor data,
// in which case the corresponding send or recv is expected to be managed
// locally by the caller.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* _HostSend(NodeOut tensor, StringPiece tensor_name, StringPiece
                send_device, int64 send_device_incarnation, StringPiece
                recv_device, const GraphDefBuilder::Options& opts);

// Receives the named tensor from send_device on recv_device.
//
// Arguments:
// * tensor_name: The name of the tensor to receive.
// * send_device: The name of the device sending the tensor.
// * send_device_incarnation: The current incarnation of send_device.
// * recv_device: The name of the device receiving the tensor.
// * opts:
//   .WithAttr("client_terminated", bool): Defaults to false.
//     If set to true, this indicates that the node was added
// to the graph as a result of a client-side feed or fetch of Tensor data,
// in which case the corresponding send or recv is expected to be managed
// locally by the caller.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// The tensor to receive.
Node* _Recv(DataType tensor_type, StringPiece tensor_name, StringPiece
            send_device, int64 send_device_incarnation, StringPiece
            recv_device, const GraphDefBuilder::Options& opts);

// Sends the named tensor from send_device to recv_device.
//
// Arguments:
// * tensor: The tensor to send.
// * tensor_name: The name of the tensor to send.
// * send_device: The name of the device sending the tensor.
// * send_device_incarnation: The current incarnation of send_device.
// * recv_device: The name of the device receiving the tensor.
// * opts:
//   .WithAttr("client_terminated", bool): Defaults to false.
//     If set to true, this indicates that the node was added
// to the graph as a result of a client-side feed or fetch of Tensor data,
// in which case the corresponding send or recv is expected to be managed
// locally by the caller.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node.
Node* _Send(NodeOut tensor, StringPiece tensor_name, StringPiece send_device,
            int64 send_device_incarnation, StringPiece recv_device, const
            GraphDefBuilder::Options& opts);

}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_OPS_SENDRECV_OPS_H_
