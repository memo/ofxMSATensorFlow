#pragma once


#include "ofMain.h"

#ifdef Success
#undef Success  // /usr/include/X11/X.h is defining this making compile fail
#endif

#ifdef Status
#undef Status   // /usr/include/X11/X.h is defining this making compile fail
#endif

#ifdef None
#undef None     // /usr/include/X11/X.h is defining this making compile fail
#endif

#ifdef Complex
#undef Complex  // /usr/include/X11/X.h is defining this making compile fail
#endif

#ifdef BadColor
#undef BadColor // /usr/include/X11/X.h is defining this making compile fail
#endif



#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
