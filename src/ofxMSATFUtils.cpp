#include "ofxMSATFUtils.h"

namespace msa {
namespace tf {

#define OFXMSATF_LOG_RETURN_ERROR(expr, msg)    TF_RETURN_IF_ERROR(logError(expr, msg))

//--------------------------------------------------------------
tensorflow::Status log_error(const tensorflow::Status& status, const string msg) {
    if(!status.ok()) ofLogError() << msg << " | " << status.ToString();
    return status;
}


//--------------------------------------------------------------
GraphDef_ptr load_graph_def(const string path, tensorflow::Env* env) {
    string of_path(ofToDataPath(path));
    GraphDef_ptr graph_def(new tensorflow::GraphDef());
    log_error( tensorflow::ReadBinaryProto(env, of_path, graph_def.get()), "Error loading graph " + of_path );
    return graph_def;
}


//--------------------------------------------------------------
Session_ptr create_session_with_graph(
        tensorflow::GraphDef& graph_def,
        const string device,
        const tensorflow::SessionOptions& session_options)
{
    Session_ptr session(NewSession(session_options));
    if( !session) { ofLogError() << "Error creating session"; return nullptr; }

    // TODO make work with r1.0
    if(!device.empty())
        tensorflow::graph::SetDefaultDevice(device, &graph_def);

    log_error(session->Create(graph_def), "Error creating graph for session");
    return session;
}


//--------------------------------------------------------------
Session_ptr create_session_with_graph(
        GraphDef_ptr pgraph_def,
        const string device,
        const tensorflow::SessionOptions& session_options)
{
    return create_session_with_graph(*pgraph_def, device, session_options);
}



//--------------------------------------------------------------
ofVec3f tensor_to_pixel_dims(const tensorflow::Tensor &t, string chmap) {
    int rank = t.shape().dims();
    vector<int> tensor_dims(rank);
    for(int i=0; i<rank; i++) tensor_dims[i] = t.dim_size(i); // useful for debugging

    // add z to end of string to top it up to length 3, this'll make it default to 1
    while(chmap.length()<3) chmap += "z";

    // which tensor dimension to use for which image xyz component
    // initially read from chmap parameter
    ofVec3f dim_indices(chmap[0]-'0', chmap[1]-'0', chmap[2]-'0');

    // if tensor rank is less than the chmap, adjust dim_indices accordingly (
    if(rank < chmap.length()) {
        if(rank == 1) {
            //  if(dim_indices)
            dim_indices.set(0, 99, 99);   // set these large so they default to 1
        } else if(rank == 2) {
            if(dim_indices[1] > dim_indices[0]) dim_indices.set(0, 1, 99);
            else dim_indices.set(1, 0, 99);
        }
    }

    ofVec3f image_dims (
                (rank > dim_indices[0] ? t.dim_size( dim_indices[0]) : 1),
        (rank > dim_indices[1] ? t.dim_size( dim_indices[1]) : 1),
        (rank > dim_indices[2] ? t.dim_size( dim_indices[2]) : 1)
      );
    return image_dims;
}


//--------------------------------------------------------------
//void get_top_scores(tensorflow::Tensor scores_tensor, int topk_count, vector<int> &out_indices, vector<float> &out_scores, string output_name) {
    //    tensorflow::GraphDefBuilder b;
    //    tensorflow::ops::TopKV2(tensorflow::ops::Const(scores_tensor, b.opts()), tensorflow::ops::Const(topk_count, b.opts()), b.opts().WithName(output_name));

    //    // This runs the GraphDef network definition that we've just constructed, and
    //    // returns the results in the output tensors.
    //    tensorflow::GraphDef graph;
    //    b.ToGraphDef(&graph);

    //    std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
    //    session->Create(graph);

    //    // The TopK node returns two outputs, the scores and their original indices,
    //    // so we have to append :0 and :1 to specify them both.
    //    std::vector<tensorflow::Tensor> output_tensors;
    //    session->Run({}, {output_name + ":0", output_name + ":1"},{}, &output_tensors);
    //    tensor_to_vector(output_tensors[0], out_scores);
    //    tensor_to_vector(output_tensors[1], out_indices);
//}

void get_topk(const vector<float> probs, vector<int> &out_indices, vector<float> &out_values, int k) {
    // http://stackoverflow.com/questions/14902876/indices-of-the-k-largest-elements-in-an-unsorted-length-n-array
    out_indices.resize(k);
    out_values.resize(k);
    std::priority_queue<std::pair<float, int>> q;
    for (int i = 0; i < probs.size(); ++i) {
        q.push(std::pair<float, int>(probs[i], i));
    }
    for (int i = 0; i < k; ++i) {
        int ki = q.top().second;
        out_indices[i] = ki;
        out_values[i] = probs[ki];
        q.pop();
    }
}

//--------------------------------------------------------------
bool read_labels_file(string file_name, vector<string>& result) {
    std::ifstream file(file_name);
    if (!file) {
        ofLogError() <<"ReadLabelsFile: " << file_name << " not found.";
        return false;
    }

    result.clear();
    string line;
    while (std::getline(file, line)) {
        result.push_back(line);
    }
    const int padding = 16;
    while (result.size() % padding) {
        result.emplace_back();
    }
    return true;
}


}
}
