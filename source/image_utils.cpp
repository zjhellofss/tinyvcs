//
// Created by fss on 22-6-13.
//
#include "image_utils.h"

void letterbox(const cv::Mat &image, cv::Mat &out_image,
               const cv::Size &new_shape,
               const cv::Scalar &color,
               bool auto_,
               bool scale_fill,
               bool scale_up,
               int stride) {
  cv::Size shape = image.size();
  float r = std::min((float) new_shape.height / (float) shape.height,
                     (float) new_shape.width / (float) shape.width);
  if (!scale_up)
    r = std::min(r, 1.0f);

  float ratio[2]{r, r};
  int new_unpad[2]{(int) std::round((float) shape.width * r),
                   (int) std::round((float) shape.height * r)};

  auto dw = (float) (new_shape.width - new_unpad[0]);
  auto dh = (float) (new_shape.height - new_unpad[1]);

  if (auto_) {
    dw = (float) ((int) dw % stride);
    dh = (float) ((int) dh % stride);
  } else if (scale_fill) {
    dw = 0.0f;
    dh = 0.0f;
    new_unpad[0] = new_shape.width;
    new_unpad[1] = new_shape.height;
    ratio[0] = (float) new_shape.width / (float) shape.width;
    ratio[1] = (float) new_shape.height / (float) shape.height;
  }

  dw /= 2.0f;
  dh /= 2.0f;

  if (shape.width != new_unpad[0] && shape.height != new_unpad[1]) {
    cv::resize(image, out_image, cv::Size(new_unpad[0], new_unpad[1]));
  }

  int top = int(std::round(dh - 0.1f));
  int bottom = int(std::round(dh + 0.1f));
  int left = int(std::round(dw - 0.1f));
  int right = int(std::round(dw + 0.1f));
  cv::copyMakeBorder(out_image, out_image, top, bottom, left, right, cv::BORDER_CONSTANT, color);
}

size_t vectorProduct(const std::vector<int64_t> &vector) {
  if (vector.empty())
    return 0;

  size_t product = 1;
  for (const auto &element : vector)
    product *= element;

  return product;
}

void scaleCoords(const cv::Size &image_shape, cv::Rect &coords, const cv::Size &image_original_shape) {
  float gain = std::min((float) image_shape.height / (float) image_original_shape.height,
                        (float) image_shape.width / (float) image_original_shape.width);

  int pad[2] = {(int) (((float) image_shape.width - (float) image_original_shape.width * gain) / 2.0f),
                (int) (((float) image_shape.height - (float) image_original_shape.height * gain) / 2.0f)};

  coords.x = (int) std::round(((float) (coords.x - pad[0]) / gain));
  coords.y = (int) std::round(((float) (coords.y - pad[1]) / gain));

  coords.width = (int) std::round(((float) coords.width / gain));
  coords.height = (int) std::round(((float) coords.height / gain));

  // // clip coords, should be modified for width and height
  // coords.x = utils::clip(coords.x, 0, image_original_shape.width);
  // coords.y = utils::clip(coords.y, 0, image_original_shape.height);
  // coords.width = utils::clip(coords.width, 0, image_original_shape.width);
  // coords.height = utils::clip(coords.height, 0, image_original_shape.height);
}
