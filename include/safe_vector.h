//
// Created by fss on 22-6-6.
//

#ifndef TINYVCS_INCLUDE_SAFE_VECTOR_H_
#define TINYVCS_INCLUDE_SAFE_VECTOR_H_

#include <mutex>
#include <optional>
#include <deque>
#include <condition_variable>

template<typename T>
class SynchronizedVector {
 public:
  explicit SynchronizedVector(int max_size = 1000) : max_size_(max_size) {
  }

  SynchronizedVector(const SynchronizedVector &synchronized_vector);

  void Push(const T &val);

  std::optional<T> Pop();

 private:
  std::mutex guard_;
  std::condition_variable cv;
  std::deque<T> vec_;
  int max_size_ = -1;
};
template<typename T>
SynchronizedVector<T>::SynchronizedVector(const SynchronizedVector &synchronized_vector) {
  {
    std::unique_lock<std::mutex> lock(guard_);
    auto copy_data = synchronized_vector.vec_;
    int size = copy_data.size();
    for (int i = 0; i < size; ++i) {
      this->vec_.push_back(copy_data.at(i));
    }
    this->max_size_ = synchronized_vector.max_size_;
  }
}

template<typename T>
void SynchronizedVector<T>::Push(const T &val) {
  std::unique_lock<std::mutex> lock(guard_);
  cv.wait(lock, [this]() {
    return this->vec_.size() < max_size_;
  });
  vec_.push_back(val);
}

template<typename T>
std::optional<T> SynchronizedVector<T>::Pop() {
  {
    std::unique_lock<std::mutex> lock(guard_);
    if (this->vec_.empty()) {
      return std::nullopt;
    } else {
      const T &val = vec_.front();
      vec_.pop_front();
      cv.notify_all();
      return val;
    }
  }
}

#endif //TINYVCS_INCLUDE_SAFE_VECTOR_H_
