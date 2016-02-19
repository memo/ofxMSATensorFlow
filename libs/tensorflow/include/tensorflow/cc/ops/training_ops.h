// This file is MACHINE GENERATED! Do not edit.

#ifndef TENSORFLOW_CC_OPS_TRAINING_OPS_H_
#define TENSORFLOW_CC_OPS_TRAINING_OPS_H_

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


// Update '*var' according to the adagrad scheme.
//
// accum += grad * grad
// var -= lr * grad * (1 / sqrt(accum))
//
// Arguments:
// * var: Should be from a Variable().
// * accum: Should be from a Variable().
// * lr: Scaling factor. Must be a scalar.
// * grad: The gradient.
// * opts:
//   .WithAttr("use_locking", bool): Defaults to false.
//     If True, updating of the var and accum tensors will be protected by
// a lock; otherwise the behavior is undefined, but may exhibit less contention.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Same as "var".
Node* ApplyAdagrad(NodeOut var, NodeOut accum, NodeOut lr, NodeOut grad, const
                   GraphDefBuilder::Options& opts);

// Update '*var' according to the Adam algorithm.
//
// lr_t <- learning_rate * sqrt(1 - beta2^t) / (1 - beta1^t)
// m_t <- beta1 * m_{t-1} + (1 - beta1) * g_t
// v_t <- beta2 * v_{t-1} + (1 - beta2) * g_t * g_t
// variable <- variable - lr_t * m_t / (sqrt(v_t) + epsilon)
//
// Arguments:
// * var: Should be from a Variable().
// * m: Should be from a Variable().
// * v: Should be from a Variable().
// * beta1_power: Must be a scalar.
// * beta2_power: Must be a scalar.
// * lr: Scaling factor. Must be a scalar.
// * beta1: Momentum factor. Must be a scalar.
// * beta2: Momentum factor. Must be a scalar.
// * epsilon: Ridge term. Must be a scalar.
// * grad: The gradient.
// * opts:
//   .WithAttr("use_locking", bool): Defaults to false.
//     If True, updating of the var, m, and v tensors will be protected by
// a lock; otherwise the behavior is undefined, but may exhibit less contention.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Same as "var".
Node* ApplyAdam(NodeOut var, NodeOut m, NodeOut v, NodeOut beta1_power, NodeOut
                beta2_power, NodeOut lr, NodeOut beta1, NodeOut beta2, NodeOut
                epsilon, NodeOut grad, const GraphDefBuilder::Options& opts);

// Update '*var' according to the Ftrl-proximal scheme.
//
// accum_new = accum + grad * grad
// linear += grad + (accum_new^(-lr_power) - accum^(-lr_power)) / lr * var
// quadratic = 1.0 / (accum_new^(lr_power) * lr) + 2 * l2
// var = (sign(linear) * l1 - linear) / quadratic if |linear| > l1 else 0.0
// accum = accum_new
//
// Arguments:
// * var: Should be from a Variable().
// * accum: Should be from a Variable().
// * linear: Should be from a Variable().
// * grad: The gradient.
// * lr: Scaling factor. Must be a scalar.
// * l1: Scaling factor. Must be a scalar.
// * l2: Scaling factor. Must be a scalar.
// * lr_power: Scaling factor. Must be a scalar.
// * opts:
//   .WithAttr("use_locking", bool): Defaults to false.
//     If True, updating of the var and accum tensors will be protected by
// a lock; otherwise the behavior is undefined, but may exhibit less contention.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Same as "var".
Node* ApplyFtrl(NodeOut var, NodeOut accum, NodeOut linear, NodeOut grad,
                NodeOut lr, NodeOut l1, NodeOut l2, NodeOut lr_power, const
                GraphDefBuilder::Options& opts);

// Update '*var' by subtracting 'alpha' * 'delta' from it.
//
// Arguments:
// * var: Should be from a Variable().
// * alpha: Scaling factor. Must be a scalar.
// * delta: The change.
// * opts:
//   .WithAttr("use_locking", bool): Defaults to false.
//     If True, the subtraction will be protected by a lock;
// otherwise the behavior is undefined, but may exhibit less contention.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Same as "var".
Node* ApplyGradientDescent(NodeOut var, NodeOut alpha, NodeOut delta, const
                           GraphDefBuilder::Options& opts);

// Update '*var' according to the momentum scheme.
//
// accum = accum * momentum + grad
// var -= lr * accum
//
// Arguments:
// * var: Should be from a Variable().
// * accum: Should be from a Variable().
// * lr: Scaling factor. Must be a scalar.
// * grad: The gradient.
// * momentum: Momentum. Must be a scalar.
// * opts:
//   .WithAttr("use_locking", bool): Defaults to false.
//     If True, updating of the var and accum tensors will be protected by
// a lock; otherwise the behavior is undefined, but may exhibit less contention.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Same as "var".
Node* ApplyMomentum(NodeOut var, NodeOut accum, NodeOut lr, NodeOut grad,
                    NodeOut momentum, const GraphDefBuilder::Options& opts);

// Update '*var' according to the RMSProp algorithm.
//
// mean_square = decay * mean_square + (1-decay) * gradient ** 2
// Delta = learning_rate * gradient / sqrt(mean_square + epsilon)
//
// ms <- rho * ms_{t-1} + (1-rho) * grad * grad
// mom <- momentum * mom_{t-1} + lr * grad / sqrt(ms + epsilon)
// var <- var - mom
//
// Arguments:
// * var: Should be from a Variable().
// * ms: Should be from a Variable().
// * mom: Should be from a Variable().
// * lr: Scaling factor. Must be a scalar.
// * rho: Decay rate. Must be a scalar.
// * epsilon: Ridge term. Must be a scalar.
// * grad: The gradient.
// * opts:
//   .WithAttr("use_locking", bool): Defaults to false.
//     If True, updating of the var, m, and v tensors will be protected by
// a lock; otherwise the behavior is undefined, but may exhibit less contention.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Same as "var".
Node* ApplyRMSProp(NodeOut var, NodeOut ms, NodeOut mom, NodeOut lr, NodeOut
                   rho, NodeOut momentum, NodeOut epsilon, NodeOut grad, const
                   GraphDefBuilder::Options& opts);

// Update relevant entries in '*var' and '*accum' according to the adagrad scheme.
//
// That is for rows we have grad for, we update var and accum as follows:
// accum += grad * grad
// var -= lr * grad * (1 / sqrt(accum))
//
// Arguments:
// * var: Should be from a Variable().
// * accum: Should be from a Variable().
// * lr: Learning rate. Must be a scalar.
// * grad: The gradient.
// * indices: A vector of indices into the first dimension of var and accum.
// * opts:
//   .WithAttr("use_locking", bool): Defaults to false.
//     If True, updating of the var and accum tensors will be protected by
// a lock; otherwise the behavior is undefined, but may exhibit less contention.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Same as "var".
Node* SparseApplyAdagrad(NodeOut var, NodeOut accum, NodeOut lr, NodeOut grad,
                         NodeOut indices, const GraphDefBuilder::Options&
                         opts);

// Update relevant entries in '*var' according to the Ftrl-proximal scheme.
//
// That is for rows we have grad for, we update var, accum and linear as follows:
// accum_new = accum + grad * grad
// linear += grad + (accum_new^(-lr_power) - accum^(-lr_power)) / lr * var
// quadratic = 1.0 / (accum_new^(lr_power) * lr) + 2 * l2
// var = (sign(linear) * l1 - linear) / quadratic if |linear| > l1 else 0.0
// accum = accum_new
//
// Arguments:
// * var: Should be from a Variable().
// * accum: Should be from a Variable().
// * linear: Should be from a Variable().
// * grad: The gradient.
// * indices: A vector of indices into the first dimension of var and accum.
// * lr: Scaling factor. Must be a scalar.
// * l1: Scaling factor. Must be a scalar.
// * l2: Scaling factor. Must be a scalar.
// * lr_power: Scaling factor. Must be a scalar.
// * opts:
//   .WithAttr("use_locking", bool): Defaults to false.
//     If True, updating of the var and accum tensors will be protected by
// a lock; otherwise the behavior is undefined, but may exhibit less contention.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Same as "var".
Node* SparseApplyFtrl(NodeOut var, NodeOut accum, NodeOut linear, NodeOut grad,
                      NodeOut indices, NodeOut lr, NodeOut l1, NodeOut l2,
                      NodeOut lr_power, const GraphDefBuilder::Options& opts);

// Update relevant entries in '*var' and '*accum' according to the momentum scheme.
//
// That is for rows we have grad for, we update var and accum as follows:
//
// accum = accum * momentum + grad
// var -= lr * accum
//
// Arguments:
// * var: Should be from a Variable().
// * accum: Should be from a Variable().
// * lr: Learning rate. Must be a scalar.
// * grad: The gradient.
// * indices: A vector of indices into the first dimension of var and accum.
// * momentum: Momentum. Must be a scalar.
// * opts:
//   .WithAttr("use_locking", bool): Defaults to false.
//     If True, updating of the var and accum tensors will be protected by
// a lock; otherwise the behavior is undefined, but may exhibit less contention.
//   .WithName(StringPiece): Set the Node's name
//   .WithDevice(StringPiece): Set the Node's requested device
//   .WithControlInput(Node*) / .WithControlInputs({Node*, ...}):
//     Add control dependencies on the specified Node(s).
//
// Returns a pointer to the created Node, with output:
// Same as "var".
Node* SparseApplyMomentum(NodeOut var, NodeOut accum, NodeOut lr, NodeOut grad,
                          NodeOut indices, NodeOut momentum, const
                          GraphDefBuilder::Options& opts);

}  // namespace ops
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_OPS_TRAINING_OPS_H_
