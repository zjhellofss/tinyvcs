//
// Created by fss on 22-6-6.
//

#ifndef TINYVCS_INCLUDE_SAFEVEC_H_
#define TINYVCS_INCLUDE_SAFEVEC_H_

#include <mutex>
#include <optional>
#include <condition_variable>

#include <mutex>
#include <queue>
#include <atomic>
#include <condition_variable>
#include <chrono>

class SynchronizedVectorException : public std::exception {
 public:
  explicit SynchronizedVectorException(const char *message) {
    message_ = message;
  }

 private:
  const char *message_ = nullptr;
};

template<class T>
class SynchronizedVector {
 public:
  explicit SynchronizedVector(int stack_size = 1000) : stack_size_(stack_size) {
  }

  void Push(const T &elem);
  T Pop();
  bool Empty();
  bool Full();
  size_t Size();
 private:
  std::mutex mutex_;
  std::queue<T> elems_{};
  std::atomic<int> stack_size_;
  std::condition_variable cond_;
};

template<class T>
void SynchronizedVector<T>::Push(const T &elem) {
  auto lock = std::unique_lock<std::mutex>(this->mutex_);

  this->cond_.wait(lock, [&]() {
    return this->elems_.size() < stack_size_;
  });

  this->elems_.push(elem);
  this->cond_.notify_one();
}

template<class T>
T SynchronizedVector<T>::Pop() {
  using namespace std::chrono_literals;
  auto lock = std::unique_lock<std::mutex>(this->mutex_);
  auto now = std::chrono::system_clock::now();
  bool wait_success = this->cond_.wait_until(lock, now + 5 * 1s, [&]() {
    return this->elems_.size() > 0;
  });
  if (wait_success) {
    T elem = this->elems_.front();
    this->elems_.pop();
    this->cond_.notify_one();
    return elem;
  } else {
    throw SynchronizedVectorException("SynchronizedVector wait pop time out");
  }
}

template<class T>
bool SynchronizedVector<T>::Empty() {
  bool is_empty = false;
  this->mutex_.lock();
  if (this->elems_.empty()) {
    is_empty = true;
  }
  this->mutex_.unlock();
  return is_empty;
}

template<class T>
bool SynchronizedVector<T>::Full() {
  this->mutex_.lock();
  if (this->elems_.size() >= this->stack_size_) {
    this->mutex_.unlock();
    return true;
  } else {
    // elems.size() < this->stack_size_
    this->mutex_.unlock();
    return false;
  }
}
template<class T>
size_t SynchronizedVector<T>::Size() {
  auto lock = std::unique_lock<std::mutex>(this->mutex_);
  size_t sz = this->elems_.size();
  return sz;
}

#endif //TINYVCS_INCLUDE_SAFEVEC_H_
