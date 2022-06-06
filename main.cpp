#include <iostream>
#include <vector>
#include <thread>
#include <deque>
#include <mutex>
#include <optional>
#include <condition_variable>
#include <utility>
#include <atomic>

#include "frame.h"
#include "safe_vector.h"

std::mutex global_mutex;

class ChainOperator {
 protected:
  virtual void Process() = 0;
  virtual void Run() = 0;
};

struct Image {
};

struct DetectInfo {
};

class PreprocessingOP {
 public:
  virtual Image Apply(const Image &image) const = 0;
  virtual ~PreprocessingOP() {};
};

class ResizeOP : public PreprocessingOP {
  Image Apply(const Image &image) const override {
	return image;
  }
};

class NormalizeOP : public PreprocessingOP {
  Image Apply(const Image &image) const override {
	return image;
  }
};

class VideoStream : public ChainOperator {
 public:
  explicit VideoStream(int stream_idx) : stream_idx_(stream_idx) {
  }

  ~VideoStream() {
	for (std::thread &t: threads_) {
	  if (t.joinable())
		t.join();
	}
  }

  void Run() override {
	std::thread t1([this]() {
	  this->Process();
	});
	threads_.push_back(std::move(t1));
  }

  bool Open(const std::string &rtsp) {
	printf("open successfully\n");
	bool has_open = true;
	if (!has_open) {

	}
	return true;
  }

  std::optional<Frame> GetFrame() {
	std::optional<Frame> ops = frames_.Pop();
	return ops;
  }

 private:
  void Process() override {
	while (true) {
//	  Frame f(stream_idx_, pts_);
//	  frames_.Push(f);
	  this->pts_ += 1;
	}
  }

 private:
  SynchronizedVector<Frame> frames_;
  std::vector<std::thread> threads_;
  int stream_idx_ = -1;
  int64_t pts_ = 0;
};

class ImageGenerator : public ChainOperator {
 public:
  explicit ImageGenerator(const std::vector<std::shared_ptr<PreprocessingOP>> &ops,
						  const std::vector<std::string> &rtsps)
	  : ops_(std::move(ops)) {
	int num = 0;
	for (const std::string &rtsp: rtsps) {
	  std::shared_ptr<VideoStream> video_stream = std::make_shared<VideoStream>(num);
	  bool success = video_stream->Open(rtsp);
	  if (!success) {
		continue;
	  } else {
		num += 1;
		this->streams_.push_back(video_stream);
	  }
	}
  }

  ~ImageGenerator() {
	for (std::thread &t: threads_) {
	  if (t.joinable())
		t.join();
	}

  }

  std::optional<Image> GetImage() {
	std::optional<Image> image_ops = this->images_.Pop();
	return image_ops;
  }

  void Run() override {
	std::thread t1([this]() {
	  this->Process();
	});
	threads_.push_back(std::move(t1));
  }

 private:
  void Process() override {
	for (auto &stream: streams_) {
	  stream->Run();
	}
	while (true) {
	  for (int i = 0; i < streams_.size(); ++i) {
		auto stream = streams_.at(i);
		std::optional<Frame> frame_ops = stream->GetFrame();
		if (frame_ops.has_value()) {
		  Frame frame = frame_ops.value();
		  const Image &image = ConvertImage(frame_ops.value());
		  images_.Push(image);
		  global_mutex.lock();
		  printf("use frame %lld %lld,generate image\n", frame.stream_idx_, frame.pts_);
		  global_mutex.unlock();
		}
	  }
	}
  }

  Image ConvertImage(const Frame &frame) const {
	Image image;
	for (const auto &op: ops_) {
	  image = op->Apply(image);
	}
	return image;
  }

 private:
  std::vector<std::shared_ptr<VideoStream>> streams_;
  SynchronizedVector<Image> images_;
  std::vector<std::shared_ptr<PreprocessingOP>> ops_;
  std::vector<std::thread> threads_;
};

class ModelInference : public ChainOperator {
 public:

  explicit ModelInference(const std::vector<std::shared_ptr<PreprocessingOP>> &ops,
						  const std::vector<std::string> &rtsps)
	  : generator_(ops, rtsps) {

  }
  ~ModelInference() {
	for (std::thread &t: threads_) {
	  if (t.joinable())
		t.join();
	}
  }

  void Run() override {
	std::thread t1([this]() {
	  this->Process();
	});
	threads_.push_back(std::move(t1));
  }

 private:
  void Process() override {
	generator_.Run();
	while (true) {
	  std::optional<Image> image = generator_.GetImage();
	  if (image.has_value()) {
		const DetectInfo &info = Detect(image.value());
//		infos_.Push(info);
		global_mutex.lock();
		printf("use image,generate detection info\n");
		global_mutex.unlock();
	  }
	}
  }

  DetectInfo Detect(const Image &image) {
	DetectInfo info;
	return info;
  }

 private:
  ImageGenerator generator_;
  SynchronizedVector<DetectInfo> infos_;
  std::vector<std::thread> threads_;
};

//int main() {
//  std::vector<std::shared_ptr<PreprocessingOP>> ops;
//  ops.push_back(std::make_shared<ResizeOP>());
//  ops.push_back(std::make_shared<NormalizeOP>());
//  ModelInference inference(ops, {"331", "332", "333", "334"});
//  inference.Run();
//  return 0;
//}
