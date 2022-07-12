#ifndef _ST_HPC_PPL_CV_CUDA_MEMORY_POOL_H_
#define _ST_HPC_PPL_CV_CUDA_MEMORY_POOL_H_

#include <forward_list>
#include <mutex>

#include "boost/core/noncopyable.hpp"

#define PITCH_GRANULARITY 512
#define PITCH_SHIFT 9

struct GpuMemoryBlock {
  unsigned char *data;
  size_t offset;
  size_t size;
  size_t pitch;
};

 class GpuMemoryPool: private boost::noncopyable{
 public:
  GpuMemoryPool();
  ~GpuMemoryPool();

  bool is_activated() const {
    return (memory_pool_ != nullptr);
  }

  void MallocMemoryPool(size_t size);
  void FreeMemoryPool();
  void Malloc1DBlock(size_t size, GpuMemoryBlock &memory_block);
  void FreeMemoryBlock(GpuMemoryBlock &memory_block);

 private:
  unsigned char *memory_pool_;
  size_t capability_;
  std::forward_list<GpuMemoryBlock> memory_blocks_;
  std::mutex host_mutex_;
};

#endif  // _ST_HPC_PPL_CV_CUDA_MEMORY_POOL_H_