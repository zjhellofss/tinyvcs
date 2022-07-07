//
// Created by fss on 22-7-7.
//

#ifndef TINYVCS_INCLUDE_SYNC_QUEUE_H_
#define TINYVCS_INCLUDE_SYNC_QUEUE_H_
#include <deque>
#include <mutex>
#include <condition_variable>
#include <chrono>

#include <boost/core/noncopyable.hpp>

template<typename T, int capacity, int wait_time_out = 1>
class SynchronizedQueue : boost::noncopyable {
 public:
  bool push(const T &value) {
    using namespace std::chrono_literals;
    std::unique_lock<std::mutex> lk(this->mutex_);
    auto now = std::chrono::system_clock::now();
    if (!cond_.wait_until(lk, now + wait_time_out_ * 100ms, [&]() {
      bool is_full = this->elem_queue_.size() >= this->capacity_;
      return !is_full;
    })) {
      return false;
    } else {
      this->elem_queue_.push_back(value);
      cond_.notify_one();
      return true;
    }
  }

  bool pop(T &value) {
    using namespace std::chrono_literals;
    std::unique_lock<std::mutex> lk(this->mutex_);
    auto now = std::chrono::system_clock::now();
    if (!cond_.wait_until(lk, now + wait_time_out_ * 100ms, [&]() {
      bool is_empty = this->elem_queue_.empty();
      return !is_empty;
    })) {
      return false;
    } else {
      value = this->elem_queue_.front();
      this->elem_queue_.pop_front();
      cond_.notify_one();
      return true;
    }
  }

  size_t write_available() {
    std::unique_lock<std::mutex> lk(this->mutex_);
    return this->capacity_ - this->elem_queue_.size();
  }

 private:
  int wait_time_out_ = wait_time_out;// 100ms time out
  int capacity_ = capacity;
  std::deque<T> elem_queue_;
  std::mutex mutex_;
  std::condition_variable cond_;
};
#endif //TINYVCS_INCLUDE_SYNC_QUEUE_H_
