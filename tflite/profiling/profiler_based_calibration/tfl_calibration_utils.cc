/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tflite/profiling/profiler_based_calibration/tfl_calibration_utils.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "Eigen/Core"  // from @eigen_archive
#include "tflite/c/common.h"
#include "tflite/core/subgraph.h"
#include "tflite/interpreter.h"
#include "tflite/profiling/memory_info.h"
#include "tflite/profiling/profiler_based_calibration/tfl_dynamic_histogram.h"
#include "tflite/profiling/profiler_based_calibration/tfl_tensor_stats_profiler.h"
#include "tflite/stateful_error_reporter.h"

namespace odml {
namespace {

void UpdateRangeMap(absl::string_view tensor_name, float current_min,
                    float current_max,
                    std::map<std::string, Range>& tensor_stats) {
  auto [it, inserted] = tensor_stats.insert(
      {std::string(tensor_name), Range{current_min, current_max}});
  if (!inserted) {
    it->second.min = std::min(it->second.min, current_min);
    it->second.max = std::max(it->second.max, current_max);
  }
}

template <typename T>
void UpdateFloatTensorStats(const T* tensor_buffer, int num_elements,
                            absl::string_view tensor_name,
                            const CalibrationOptions& options,
                            std::map<std::string, Range>& tensor_stats) {
  constexpr float kInf = std::numeric_limits<float>::infinity();
  float filtered_min = kInf;
  float filtered_max = -kInf;
  float raw_min = kInf;
  float raw_max = -kInf;
  bool has_non_nan = false;

  for (int i = 0; i < num_elements; ++i) {
    const float val = static_cast<float>(tensor_buffer[i]);
    if (std::isnan(val)) {
      continue;
    }
    has_non_nan = true;
    raw_min = std::min(raw_min, val);
    raw_max = std::max(raw_max, val);
    if (val > options.valid_min) {
      filtered_min = std::min(filtered_min, val);
    }
    if (val < options.valid_max) {
      filtered_max = std::max(filtered_max, val);
    }
  }
  if (!has_non_nan) {
    return;
  }
  const float current_min = (filtered_min == kInf) ? raw_min : filtered_min;
  const float current_max = (filtered_max == -kInf) ? raw_max : filtered_max;
  UpdateRangeMap(tensor_name, current_min, current_max, tensor_stats);
}

template <typename T>
void UpdateIntTensorStats(const T* tensor_buffer, int num_elements,
                          absl::string_view tensor_name,
                          std::map<std::string, Range>& tensor_stats) {
  const Eigen::Map<const Eigen::VectorX<T>> vec(tensor_buffer, num_elements);
  UpdateRangeMap(tensor_name, static_cast<float>(vec.minCoeff()),
                 static_cast<float>(vec.maxCoeff()), tensor_stats);
}

struct MaskInfo {
  int mask_seq_len = 0;
  int valid_seq_len = 0;
};

// Detects active token count from a 2D [1, seq_len] mask input tensor if
// present so that token-wise activation statistics and histograms exclude
// mask == 0 right-padded positions.
MaskInfo DetectMaskInfo(const tflite::Subgraph* subgraph) {
  if (subgraph == nullptr) {
    return {};
  }
  for (int inp_idx : subgraph->inputs()) {
    const TfLiteTensor* inp = subgraph->tensor(inp_idx);
    if (inp == nullptr || inp->name == nullptr || inp->data.raw == nullptr ||
        inp->dims == nullptr || inp->dims->size != 2 ||
        inp->dims->data[0] != 1 || inp->dims->data[1] <= 1 ||
        !absl::StrContains(absl::string_view(inp->name), "mask")) {
      continue;
    }
    const int seq_len = inp->dims->data[1];
    int valid = 0;
    if (inp->type == kTfLiteFloat32) {
      valid = std::count_if(inp->data.f, inp->data.f + seq_len,
                            [](float v) { return v != 0.0f; });
    } else if (inp->type == kTfLiteInt32) {
      valid = std::count_if(inp->data.i32, inp->data.i32 + seq_len,
                            [](int32_t v) { return v != 0; });
    }
    if (valid > 0) {
      return {.mask_seq_len = seq_len, .valid_seq_len = valid};
    }
  }
  return {};
}

int GetEffectiveNumElements(const TfLiteTensor& tensor, int total_elements,
                            MaskInfo mask_info) {
  if (mask_info.valid_seq_len <= 0 ||
      mask_info.valid_seq_len >= mask_info.mask_seq_len ||
      tensor.dims == nullptr || tensor.dims->size < 3 ||
      tensor.dims->data[0] != 1 ||
      tensor.dims->data[1] != mask_info.mask_seq_len ||
      total_elements % mask_info.mask_seq_len != 0) {
    return total_elements;
  }
  return (total_elements / mask_info.mask_seq_len) * mask_info.valid_seq_len;
}

}  // namespace

CalibrationSession::CalibrationSession(CalibrationOptions options)
    : options_(std::move(options)) {}

bool CalibrationSession::ShouldCollectHistogram(
    absl::string_view tensor_name) const {
  return options_.collect_histogram &&
         (!options_.histogram_tensor_names.has_value() ||
          options_.histogram_tensor_names->count(std::string(tensor_name)) > 0);
}

DynamicHistogram& CalibrationSession::GetOrCreateHistogram(
    absl::string_view tensor_name) {
  return histograms_.try_emplace(tensor_name, options_.num_bins).first->second;
}

absl::StatusOr<std::map<std::string, Range>> CalibrationSession::Invoke(
    tflite::Interpreter* interpreter, int subgraph_index,
    tflite::StatefulErrorReporter* reporter) {
  tflite::Subgraph* main_subgraph = interpreter->subgraph(subgraph_index);
  const MaskInfo mask_info = DetectMaskInfo(main_subgraph);

  // Use a callback to capture tensor stats during operator invocation. The
  // results are stored in a map of tensor name to min/max stat value.
  std::map<std::string, Range> tensor_stats;
  auto calc_tensor_stats = [&](const TfLiteTensor* tensor) {
    // Skip constant, empty, or unallocated tensors.
    if (tensor->allocation_type == kTfLiteMmapRo ||
        tensor->allocation_type == kTfLitePersistentRo ||
        tensor->data.raw == nullptr || tensor->name == nullptr ||
        tensor->bytes == 0) {
      return;
    }
    const absl::string_view tensor_name(tensor->name);
    const bool collect_hist = ShouldCollectHistogram(tensor_name);
    auto update_float = [&](const auto* data) {
      const int total_n = tensor->bytes / sizeof(*data);
      const int n = GetEffectiveNumElements(*tensor, total_n, mask_info);
      UpdateFloatTensorStats(data, n, tensor_name, options_, tensor_stats);
      if (collect_hist) {
        GetOrCreateHistogram(tensor_name)
            .Add(data, n, options_.valid_min, options_.valid_max);
      }
    };
    switch (tensor->type) {
      case kTfLiteFloat32:
        update_float(tensor->data.f);
        break;
      case kTfLiteFloat16:
        update_float(reinterpret_cast<const Eigen::half*>(tensor->data.f16));
        break;
      case kTfLiteBFloat16:
        update_float(
            reinterpret_cast<const Eigen::bfloat16*>(tensor->data.bf16));
        break;
      case kTfLiteInt32: {
        const int total_n = tensor->bytes / sizeof(int32_t);
        const int n = GetEffectiveNumElements(*tensor, total_n, mask_info);
        UpdateIntTensorStats<int32_t>(tensor->data.i32, n, tensor_name,
                                      tensor_stats);
        if (collect_hist) {
          GetOrCreateHistogram(tensor_name).Add(tensor->data.i32, n);
        }
        break;
      }
      default:
        break;
    }
  };

  auto profiler =
      std::make_unique<TensorStatsProfiler>(*interpreter, calc_tensor_stats);
  interpreter->SetProfiler(profiler.get());

  const TfLiteStatus invoke_status = main_subgraph->Invoke();

  // Reset the profiler to avoid dangling pointer issues when interpreter is
  // reused.
  interpreter->SetProfiler(nullptr);
  for (size_t i = 0; i < interpreter->subgraphs_size(); ++i) {
    interpreter->subgraph(i)->SetProfiler(nullptr, i);
  }

  if (invoke_status != kTfLiteOk) {
    if (reporter != nullptr) {
      return absl::InternalError(
          absl::StrCat("InvokeWithCalibration failed: ", reporter->message()));
    }
    return absl::InternalError("InvokeWithCalibration failed");
  }

  return tensor_stats;
}

std::map<std::string, TensorHistogram> CalibrationSession::GetHistograms()
    const {
  std::map<std::string, TensorHistogram> result;
  for (const auto& [name, hist] : histograms_) {
    if (!hist.initialized()) {
      continue;
    }
    result.emplace(name, TensorHistogram{
                             .counts = hist.counts(),
                             .lower_bound = hist.lower_bound(),
                             .bin_width = hist.bin_width(),
                             .min = hist.global_min(),
                             .max = hist.global_max(),
                         });
  }
  return result;
}

void CalibrationSession::Reset() { histograms_.clear(); }

absl::StatusOr<std::map<std::string, Range>> InvokeWithCalibration(
    tflite::Interpreter* interpreter, int subgraph_index,
    tflite::StatefulErrorReporter* reporter) {
  CalibrationSession session;
  return session.Invoke(interpreter, subgraph_index, reporter);
}

tflite::profiling::memory::MemoryUsage GetMemoryUsage() {
  return tflite::profiling::memory::GetMemoryUsage();
}

}  // namespace odml
