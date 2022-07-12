
#include "cuda/memory_pool.h"
#include "glog/logging.h"
#include "cuda_runtime_api.h"

inline int roundUp(int total, int grain, int shift) {
  return ((total + grain - 1) >> shift) << shift;
}

GpuMemoryPool::GpuMemoryPool() {
  memory_pool_ = nullptr;
}

GpuMemoryPool::~GpuMemoryPool() {
  if (memory_pool_ != nullptr) {
    cudaError_t code = cudaFree(memory_pool_);
    if (code != cudaSuccess) {
      LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    }
    memory_blocks_.clear();
  }
}

void GpuMemoryPool::MallocMemoryPool(size_t size) {
  cudaError_t code = cudaMalloc((void **) &memory_pool_, size);
  if (code != cudaSuccess) {
    LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
  }

  capability_ = size;
}

void GpuMemoryPool::FreeMemoryPool() {
  if (memory_pool_ != nullptr) {
    cudaError_t code = cudaFree(memory_pool_);
    if (code != cudaSuccess) {
      LOG(ERROR) << "CUDA error: " << cudaGetErrorString(code);
    }

    capability_ = 0;
    memory_pool_ = nullptr;
  }
}

void GpuMemoryPool::Malloc1DBlock(size_t size, GpuMemoryBlock &memory_block) {
  if (memory_blocks_.empty()) {
    if (size <= capability_) {
      memory_block.data = memory_pool_;
      memory_block.offset = 0;
      memory_block.size = size;
      memory_block.pitch = 0;

      host_mutex_.lock();
      auto current = memory_blocks_.before_begin();
      memory_blocks_.insert_after(current, memory_block);
      host_mutex_.unlock();
    } else {
      LOG(ERROR) << "Cuda Memory Pool error: insufficient space.";
    }

    return;
  }

  auto previous = memory_blocks_.begin();
  auto current = memory_blocks_.begin();
  ++current;
  if (current == memory_blocks_.end()) {
    size_t hollow_begin = roundUp((previous->offset + previous->size),
                                  PITCH_GRANULARITY, PITCH_SHIFT);
    if (hollow_begin + size <= capability_) {
      memory_block.data = memory_pool_;
      memory_block.offset = hollow_begin;
      memory_block.size = size;
      memory_block.pitch = 0;

      host_mutex_.lock();
      memory_blocks_.insert_after(previous, memory_block);
      host_mutex_.unlock();
    } else {
      LOG(ERROR) << "Cuda Memory Pool error: insufficient space.";
    }
    return;
  }

  while (current != memory_blocks_.end()) {
    size_t hollow_begin = roundUp((previous->offset + previous->size),
                                  PITCH_GRANULARITY, PITCH_SHIFT);
    if (hollow_begin + size <= current->offset) {
      memory_block.data = memory_pool_;
      memory_block.offset = hollow_begin;
      memory_block.size = size;
      memory_block.pitch = 0;

      host_mutex_.lock();
      memory_blocks_.insert_after(previous, memory_block);
      host_mutex_.unlock();
      break;
    }
    ++current;
  }

  if (current == memory_blocks_.end()) {
    LOG(ERROR) << "Cuda Memory Pool error: insufficient space.";
  }
}


void GpuMemoryPool::FreeMemoryBlock(GpuMemoryBlock &memory_block) {
  if (memory_blocks_.empty()) {
    LOG(ERROR) << "Cuda Memory Pool error: empty pool can't contain a block.";

    return;
  }

  auto previous = memory_blocks_.before_begin();
  auto current = memory_blocks_.begin();
  while (current != memory_blocks_.end()) {
    if (current->offset == memory_block.offset) {
      host_mutex_.lock();
      memory_blocks_.erase_after(previous);
      host_mutex_.unlock();
      break;
    }

    previous = current;
    ++current;
  }

  if (current == memory_blocks_.end()) {
    LOG(ERROR) << "Cuda Memory Pool error: can't not find the memory block.";
  }
}

