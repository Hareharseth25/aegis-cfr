// A small fixed-size thread pool with a blocking parallel_for. Work items are
// independent (e.g. river-card subtrees below a chance node), so results are
// deterministic regardless of thread scheduling.
#pragma once

#include <atomic>
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace aegis {

class ThreadPool {
 public:
  explicit ThreadPool(int num_threads);
  ~ThreadPool();
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  int size() const { return static_cast<int>(workers_.size()) + 1; }

  // Runs fn(i) for i in [0, n). The calling thread participates. Exceptions
  // thrown by any task are rethrown on the calling thread. Calls from
  // different external threads are serialised; nested calls from inside a
  // task run serially on the calling worker.
  void parallel_for(int n, const std::function<void(int)>& fn);

  static int hardware_threads();

 private:
  void worker_loop();
  void run_tasks();

  std::vector<std::thread> workers_;
  std::mutex call_mu_;  // serialises parallel_for calls from different external threads
  std::mutex mu_;
  std::condition_variable cv_start_;
  std::condition_variable cv_done_;
  const std::function<void(int)>* fn_ = nullptr;
  int n_ = 0;
  std::atomic<int> next_{0};
  int active_ = 0;
  uint64_t generation_ = 0;
  bool stop_ = false;
  std::exception_ptr error_;
};

}  // namespace aegis
