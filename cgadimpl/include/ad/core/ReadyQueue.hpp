#include <memory>
#include <queue>
#include <future>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include "ad/core/graph.hpp"

namespace ag{
class readyqueue{
    std::queue<ag::Node*> q;
    std::mutex m;
    std::condition_variable cv;
    bool finished = false;

    public:
    // readyqueue() : finished(false) {}

    //pushes the element that is ready into the readyqueue if its ready to compute its gradients
    void push(Node* node){
        {
            std::lock_guard<std::mutex> lck(m);
            q.push(node);
        }
        cv.notify_one();
    }

    // pops the element out the scope once its done its job
    Node* pop(){
        std::unique_lock<std::mutex> lock(m);
        cv.wait(lock,[this]{return !q.empty()||finished;});
        if (finished && q.empty()){
            return nullptr;
        }
        Node* node = q.front();
        q.pop();
        return node;
    }

    //if there is a nullptr in the queue then the whole backward process is completed
    void shutdown(){
        {
            std::lock_guard<std::mutex> lock(m);
            finished = true;
        }
        cv.notify_all();
    }

};
}
