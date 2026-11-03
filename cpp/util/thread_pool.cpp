#include "util/thread_pool.h"

namespace aegis {

namespace {
// Nested parallel_for calls from inside a task run serially instead of
// deadlocking on the (single-generation) pool state.
thread_local bool tl_in_pool_task = false;
struct TaskScope {
  bool prev;
  TaskScope() : prev(tl_in_pool_task) { tl_in_pool_task = true; }
  ~TaskScope() { tl_in_pool_task = prev; }
};
}  // namespace

ThreadPool::ThreadPool(int num_threads) {
  if (num_threads < 1) num_threads = 1;
  for (int i = 0; i < num_threads - 1; ++i) workers_.emplace_back([this] { worker_loop(); });
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> lock(mu_);
    stop_ = true;
  }
  cv_start_.notify_all();
  for (auto& t : workers_) t.join();
}

int ThreadPool::hardware_threads() {
  const unsigned n = std::thread::hardware_concurrency();
  return n == 0 ? 1 : static_cast<int>(n);
}

void ThreadPool::run_tasks() {
  while (true) {
    const int i = next_.fetch_add(1);
    if (i >= n_) break;
    try {
      TaskScope scope;
      (*fn_)(i);
    } catch (...) {
      std::lock_guard<std::mutex> lock(mu_);
      if (!error_) error_ = std::current_exception();
    }
  }
}

void ThreadPool::worker_loop() {
  uint64_t seen = 0;
  while (true) {
    {
      std::unique_lock<std::mutex> lock(mu_);
      cv_start_.wait(lock, [&] { return stop_ || generation_ != seen; });
      if (stop_) return;
      seen = generation_;
    }
    run_tasks();
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (--active_ == 0) cv_done_.notify_all();
    }
  }
}

void ThreadPool::parallel_for(int n, const std::function<void(int)>& fn) {
  if (n <= 0) return;
  if (workers_.empty() || n == 1 || tl_in_pool_task) {
    for (int i = 0; i < n; ++i) fn(i);
    return;
  }
  std::lock_guard<std::mutex> call_lock(call_mu_);
  {
    std::lock_guard<std::mutex> lock(mu_);
    fn_ = &fn;
    n_ = n;
    next_.store(0);
    active_ = static_cast<int>(workers_.size());
    error_ = nullptr;
    ++generation_;
  }
  cv_start_.notify_all();
  run_tasks();
  std::exception_ptr err;
  {
    std::unique_lock<std::mutex> lock(mu_);
    cv_done_.wait(lock, [&] { return active_ == 0; });
    fn_ = nullptr;
    err = error_;
  }
  if (err) std::rethrow_exception(err);
}

}  // namespace aegis
