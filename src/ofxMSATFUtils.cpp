#include "ofxMSATFUtils.h"

namespace msa {
namespace tf {

//#define OFXMSATF_LOG_RETURN_ERROR(expr, msg)    TF_RETURN_IF_ERROR(logError(expr, msg))

//--------------------------------------------------------------
tensorflow::Status log_error(const tensorflow::Status& status, const string msg) {
    if(!status.ok()) {
        string s = msg + " | " + status.ToString();
        ofLogError("ofxMSATensorFlow") << s;
        throw std::runtime_error(s);
    }
    return status;
}


//--------------------------------------------------------------
string missing_data_error() {
    string s;
    s += "Did you download the data files and place them in the data folder?\n";
    s += "Download from https://github.com/memo/ofxMSATensorFlow/releases\n";
    s += "More info at https://github.com/memo/ofxMSATensorFlow/wiki\n";
    return s;
}


//--------------------------------------------------------------
Session_ptr create_session(const tensorflow::SessionOptions& session_options) {
    tensorflow::Session* session;
    auto status = tensorflow::NewSession(session_options, &session);
    log_error(status, "create_session");
    return Session_ptr(session);
}


//--------------------------------------------------------------
GraphDef_ptr load_graph_def(const string path, tensorflow::Env* env) {
    string of_path(ofToDataPath(path));
    GraphDef_ptr graph_def(new tensorflow::GraphDef()); // TODO try this as normal pointer and change to shared later
    auto status = tensorflow::ReadBinaryProto(env, of_path, graph_def.get());
    log_error(status, "load_graph_def: " + of_path );
    return graph_def;
}


//--------------------------------------------------------------
void create_graph_in_session(Session_ptr session, GraphDef_ptr graph_def, const string device) {
    if(!device.empty()) tensorflow::graph::SetDefaultDevice(device, graph_def.get());
    auto status = session->Create(*graph_def);
    log_error(status, "create_graph_in_session");
}


//--------------------------------------------------------------
tensorflow::SessionOptions session_gpu_options(bool allow_growth, double per_process_gpu_memory_fraction, const tensorflow::SessionOptions& session_options_base) {
    tensorflow::SessionOptions session_options(session_options_base);
    session_options.config.mutable_gpu_options()->set_allow_growth(allow_growth);
    session_options.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(per_process_gpu_memory_fraction);
    return session_options;
}


//--------------------------------------------------------------
Session_ptr create_session_with_graph(
        GraphDef_ptr graph_def,
        const string device,
        const tensorflow::SessionOptions& session_options)
{
    Session_ptr session = create_session(session_options);
    create_graph_in_session(session, graph_def, device);
    return session;
}


//--------------------------------------------------------------
Session_ptr create_session_with_graph(
        const string graph_def_path,
        const string device,
        const tensorflow::SessionOptions& session_options)
{
    GraphDef_ptr graph_def = load_graph_def(graph_def_path);
    if(graph_def) return create_session_with_graph(graph_def, device, session_options);
    return nullptr;
}


//--------------------------------------------------------------
//Session_ptr create_session_with_graph(
//        tensorflow::GraphDef& graph_def_ref,
//        const string device,
//        const tensorflow::SessionOptions& session_options)
//{
//    Session_ptr session = create_session(session_options);
////    if( !session) { ofLogError("ofxMSATensorFlow) << "Error creating session"; return nullptr; }


//    log_error(session->Create(graph_def_ref), "Error creating graph for session");
//    return session;
//}


//--------------------------------------------------------------
vector<tensorflow::int64> tensor_to_pixel_dims(const tensorflow::Tensor &t, string chmap) {
    int rank = t.shape().dims();
    vector<tensorflow::int64> tensor_dims(rank);
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

    vector<tensorflow::int64> image_dims( {
                                              (rank > dim_indices[0] ? (int)t.dim_size( dim_indices[0]) : 1),
                                              (rank > dim_indices[1] ? (int)t.dim_size( dim_indices[1]) : 1),
                                              (rank > dim_indices[2] ? (int)t.dim_size( dim_indices[2]) : 1)
                                          });
    return image_dims;
}


//--------------------------------------------------------------
vector<tensorflow::int64> get_imagedims_for_tensorshape(const vector<tensorflow::int64>& tensorshape, bool shape_includes_batch) {
    tensorflow::int64 h_index = shape_includes_batch ? 1 : 0;   // index in shape for image height
    tensorflow::int64 w_index = shape_includes_batch ? 2 : 1;   // index in shape for image width
    tensorflow::int64 c_index = shape_includes_batch ? 3 : 2;   // index in shape for number of channels

    tensorflow::int64 h = h_index < tensorshape.size() ? tensorshape[h_index] : 1; // value for image height
    tensorflow::int64 w = w_index < tensorshape.size() ? tensorshape[w_index] : 1; // value for image width
    tensorflow::int64 c = c_index < tensorshape.size() ? tensorshape[c_index] : 1; // value for image height
    return {w, h, c};
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
        ofLogError("ofxMSATensorFlow") <<"read_labels_file: " << file_name << " not found.";
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


//--------------------------------------------------------------
vector<string> get_file_list(string model_dir, int max_count, string ext, bool do_sort) {
    ofLogVerbose("ofxMSATensorFlow")  << "get_file_list:" << model_dir << ", max_count:" << max_count << ", ext:" << ext;

    ofDirectory dir;
    dir.allowExt(ext);
    dir.listDir(model_dir);
    if(dir.size()==0) {
        throw std::runtime_error("get_file_list: no models in " + model_dir);
    }
    vector<string> filenames;
    for(int i=0; i<dir.getFiles().size(); i++) filenames.push_back(dir.getName(i));
    if(do_sort) sort(filenames.begin(), filenames.end());

    int count = filenames.size();
    ofLogVerbose("ofxMSATensorFlow") << count << " checkpoints found";
    if(count > max_count) {
        std::reverse(filenames.begin(),filenames.end());
        vector<string> new_filenames;
        ofLogVerbose("ofxMSATensorFlow") << "too many, using only " << max_count;
        for(int i=0; i<max_count; i++) {
            int index = round(ofMap(i, 0, max_count-1, 0, count-1));
            new_filenames.push_back(filenames[index]);
            ofLogVerbose() << i << " -> " << index << " : " << new_filenames.back();
        }
        filenames = new_filenames;
        std::reverse(filenames.begin(),filenames.end());
    }
    return filenames;
}


}
}
