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

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "Eigen/Core"  // from @eigen_archive
#include "tflite/c/common.h"
#include "tflite/interpreter.h"
#include "tflite/profiling/memory_info.h"
#include "tflite/stateful_error_reporter.h"

namespace odml {
namespace {

using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::status::StatusIs;

class FakeErrorReporter : public tflite::StatefulErrorReporter {
 public:
  int Report(const char*, va_list) override { return 0; }
  std::string message() override { return "custom error message"; }
};

int64_t SumCounts(const TensorHistogram& hist) {
  int64_t total = 0;
  for (int64_t c : hist.counts) {
    total += c;
  }
  return total;
}

class TflCalibrationUtilsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    interpreter_ = std::make_unique<tflite::Interpreter>();
  }

  int AddTensor(const char* name, TfLiteType type,
                const std::vector<int>& dims) {
    int index = 0;
    CHECK_EQ(interpreter_->AddTensors(1, &index), kTfLiteOk);
    CHECK_EQ(interpreter_->SetTensorParametersReadWrite(index, type, name, dims,
                                                        TfLiteQuantization()),
             kTfLiteOk);
    return index;
  }

  template <typename T, size_t N>
  void PopulateTensor(int index, const T (&data)[N]) {
    std::memcpy(interpreter_->tensor(index)->data.raw, data, sizeof(data));
  }

  void AddNoOpNode(const std::vector<int>& inputs,
                   const std::vector<int>& outputs = {}) {
    static TfLiteRegistration reg = {
        .invoke = [](TfLiteContext*, TfLiteNode*) { return kTfLiteOk; }};
    CHECK_EQ(interpreter_->AddNodeWithParameters(
                 inputs, outputs, /*init_data=*/nullptr,
                 /*init_data_size=*/0, /*builtin_data=*/nullptr, &reg),
             kTfLiteOk);
  }

  std::unique_ptr<tflite::Interpreter> interpreter_;
};

TEST_F(TflCalibrationUtilsTest, InvokeWithCalibrationFloat) {
  AddTensor("input", kTfLiteFloat32, {1, 4});
  AddTensor("output", kTfLiteFloat32, {1, 4});
  interpreter_->SetInputs({0});
  interpreter_->SetOutputs({1});

  static int invoke_count = 0;
  invoke_count = 0;
  TfLiteRegistration reg = {
      .invoke = [](TfLiteContext*, TfLiteNode*) -> TfLiteStatus {
        ++invoke_count;
        return kTfLiteOk;
      }};
  ASSERT_EQ(interpreter_->AddNodeWithParameters(
                /*inputs=*/{0}, /*outputs=*/{1}, /*init_data=*/nullptr,
                /*init_data_size=*/0, /*builtin_data=*/nullptr,
                /*registration=*/&reg),
            kTfLiteOk);

  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  const float in[] = {1.0f, 2.0f, -1.0f, 0.0f};
  PopulateTensor(0, in);

  ASSERT_OK_AND_ASSIGN(const auto stats,
                       InvokeWithCalibration(interpreter_.get(), 0));
  EXPECT_EQ(invoke_count, 1);
  ASSERT_EQ(stats.count("input"), 1);
  EXPECT_FLOAT_EQ(stats.at("input").min, -1.0f);
  EXPECT_FLOAT_EQ(stats.at("input").max, 2.0f);
}

TEST_F(TflCalibrationUtilsTest, InvokeWithCalibrationFiltersInfNaNAndAllNaN) {
  AddTensor("input", kTfLiteFloat32, {1, 6});
  AddTensor("all_nan", kTfLiteFloat32, {1, 2});
  interpreter_->SetInputs({0, 1});
  AddNoOpNode({0, 1});
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  constexpr float kInf = std::numeric_limits<float>::infinity();
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  const float in[] = {-kInf, kInf, kNaN, -2.5f, 3.5f, -3.1e38f};
  const float nans[] = {kNaN, kNaN};
  PopulateTensor(0, in);
  PopulateTensor(1, nans);

  CalibrationSession session(CalibrationOptions{
      .collect_histogram = true,
      .num_bins = 16,
  });
  ASSERT_OK_AND_ASSIGN(const auto stats, session.Invoke(interpreter_.get(), 0));
  ASSERT_EQ(stats.count("input"), 1);
  EXPECT_FLOAT_EQ(stats.at("input").min, -2.5f);
  EXPECT_FLOAT_EQ(stats.at("input").max, 3.5f);
  EXPECT_EQ(stats.count("all_nan"), 0);

  const auto hists = session.GetHistograms();
  EXPECT_EQ(hists.count("input"), 1);
  EXPECT_EQ(hists.count("all_nan"), 0);
}

TEST_F(TflCalibrationUtilsTest, InvokeWithCalibrationAllInfFallback) {
  AddTensor("input", kTfLiteFloat32, {1, 2});
  interpreter_->SetInputs({0});
  AddNoOpNode({0});
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  constexpr float kInf = std::numeric_limits<float>::infinity();
  const float in[] = {-kInf, -kInf};
  PopulateTensor(0, in);

  ASSERT_OK_AND_ASSIGN(const auto stats,
                       InvokeWithCalibration(interpreter_.get(), 0));
  ASSERT_EQ(stats.count("input"), 1);
  EXPECT_EQ(stats.at("input").min, -kInf);
  EXPECT_EQ(stats.at("input").max, -kInf);
}

TEST_F(TflCalibrationUtilsTest, InvokeWithCalibrationFloat16AndBFloat16) {
  AddTensor("input_f16", kTfLiteFloat16, {1, 4});
  AddTensor("input_bf16", kTfLiteBFloat16, {1, 4});
  interpreter_->SetInputs({0, 1});
  AddNoOpNode({0, 1});
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  constexpr float kInf = std::numeric_limits<float>::infinity();
  const Eigen::half in_f16[] = {Eigen::half(-1.5f), Eigen::half(0.0f),
                                Eigen::half(4.0f), Eigen::half(-kInf)};
  const Eigen::bfloat16 in_bf16[] = {
      Eigen::bfloat16(-2.0f), Eigen::bfloat16(1.0f), Eigen::bfloat16(8.0f),
      Eigen::bfloat16(-kInf)};
  PopulateTensor(0, in_f16);
  PopulateTensor(1, in_bf16);

  ASSERT_OK_AND_ASSIGN(const auto stats,
                       InvokeWithCalibration(interpreter_.get(), 0));
  ASSERT_EQ(stats.count("input_f16"), 1);
  EXPECT_FLOAT_EQ(stats.at("input_f16").min, -1.5f);
  EXPECT_FLOAT_EQ(stats.at("input_f16").max, 4.0f);
  ASSERT_EQ(stats.count("input_bf16"), 1);
  EXPECT_FLOAT_EQ(stats.at("input_bf16").min, -2.0f);
  EXPECT_FLOAT_EQ(stats.at("input_bf16").max, 8.0f);
}

TEST_F(TflCalibrationUtilsTest, InvokeWithCalibrationInt32AndMultiNodeMerge) {
  AddTensor("input_i32", kTfLiteInt32, {1, 4});
  AddTensor("input_f32", kTfLiteFloat32, {1, 2});
  interpreter_->SetInputs({0, 1});
  // Two nodes consume the same tensors so UpdateRangeMap exercises both insert
  // and merge paths.
  AddNoOpNode({0, 1});
  AddNoOpNode({0, 1});
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  const int32_t in_i32[] = {10, 20, -10, 0};
  const float in_f32[] = {-3.0f, 5.0f};
  PopulateTensor(0, in_i32);
  PopulateTensor(1, in_f32);

  ASSERT_OK_AND_ASSIGN(const auto stats,
                       InvokeWithCalibration(interpreter_.get(), 0));
  ASSERT_EQ(stats.count("input_i32"), 1);
  EXPECT_FLOAT_EQ(stats.at("input_i32").min, -10.0f);
  EXPECT_FLOAT_EQ(stats.at("input_i32").max, 20.0f);
  ASSERT_EQ(stats.count("input_f32"), 1);
  EXPECT_FLOAT_EQ(stats.at("input_f32").min, -3.0f);
  EXPECT_FLOAT_EQ(stats.at("input_f32").max, 5.0f);
}

TEST_F(TflCalibrationUtilsTest, SkipsConstantTensorsAndReportsInvokeError) {
  int const_idx = 0;
  ASSERT_EQ(interpreter_->AddTensors(1, &const_idx), kTfLiteOk);
  const float weights[] = {-10.0f, 10.0f};
  ASSERT_EQ(interpreter_->SetTensorParametersReadOnly(
                const_idx, kTfLiteFloat32, "const_weights", {1, 2},
                TfLiteQuantization(), reinterpret_cast<const char*>(weights),
                sizeof(weights)),
            kTfLiteOk);
  AddNoOpNode({const_idx});
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  ASSERT_OK_AND_ASSIGN(const auto stats,
                       InvokeWithCalibration(interpreter_.get(), 0));
  EXPECT_EQ(stats.count("const_weights"), 0);

  // Create a fresh interpreter with a failing node and verify error propagation
  // with and without a reporter.
  interpreter_ = std::make_unique<tflite::Interpreter>();
  AddTensor("input", kTfLiteFloat32, {1, 2});
  interpreter_->SetInputs({0});
  static TfLiteRegistration fail_reg = {
      .invoke = [](TfLiteContext*, TfLiteNode*) { return kTfLiteError; }};
  ASSERT_EQ(interpreter_->AddNodeWithParameters({0}, {}, nullptr, 0, nullptr,
                                                &fail_reg),
            kTfLiteOk);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  EXPECT_THAT(InvokeWithCalibration(interpreter_.get(), 0),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("InvokeWithCalibration failed")));

  FakeErrorReporter reporter;
  EXPECT_THAT(
      InvokeWithCalibration(interpreter_.get(), 0, &reporter),
      StatusIs(absl::StatusCode::kInternal, HasSubstr("custom error message")));
}

TEST_F(TflCalibrationUtilsTest, GetMemoryUsage) {
  tflite::profiling::memory::MemoryUsage usage = GetMemoryUsage();
  EXPECT_GT(usage.total_allocated_bytes, 0);
}

TEST_F(TflCalibrationUtilsTest,
       CalibrationSessionAccumulatesHistogramsAcrossInvokes) {
  AddTensor("input", kTfLiteFloat32, {1, 4});
  interpreter_->SetInputs({0});
  AddNoOpNode({0});
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  CalibrationSession session(CalibrationOptions{
      .collect_histogram = true,
      .num_bins = 16,
  });

  const float in1[] = {-2.0f, -1.0f, 1.0f, 2.0f};
  PopulateTensor(0, in1);
  ASSERT_OK_AND_ASSIGN(const auto stats1,
                       session.Invoke(interpreter_.get(), 0));
  EXPECT_FLOAT_EQ(stats1.at("input").min, -2.0f);
  EXPECT_FLOAT_EQ(stats1.at("input").max, 2.0f);

  // Second batch lies strictly inside the first batch's range.
  const float in2[] = {-1.5f, 0.0f, 0.5f, 1.5f};
  PopulateTensor(0, in2);
  ASSERT_OK_AND_ASSIGN(const auto stats2,
                       session.Invoke(interpreter_.get(), 0));
  // Min/max is per-invocation, not accumulated.
  EXPECT_FLOAT_EQ(stats2.at("input").min, -1.5f);
  EXPECT_FLOAT_EQ(stats2.at("input").max, 1.5f);

  // Histograms accumulate across invocations.
  const auto hists = session.GetHistograms();
  ASSERT_EQ(hists.count("input"), 1);
  const auto& hist = hists.at("input");
  EXPECT_FLOAT_EQ(hist.min, -2.0f);
  EXPECT_FLOAT_EQ(hist.max, 2.0f);
  EXPECT_EQ(hist.counts.size(), 16);
  EXPECT_EQ(SumCounts(hist), 8);

  session.Reset();
  EXPECT_THAT(session.GetHistograms(), IsEmpty());
}

TEST_F(TflCalibrationUtilsTest,
       CalibrationSessionRespectsHistogramTensorNamesFilter) {
  AddTensor("input_a", kTfLiteFloat32, {1, 2});
  AddTensor("input_b", kTfLiteFloat32, {1, 2});
  interpreter_->SetInputs({0, 1});
  AddNoOpNode({0, 1});
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  const float in_a[] = {-1.0f, 1.0f};
  const float in_b[] = {-2.0f, 2.0f};
  PopulateTensor(0, in_a);
  PopulateTensor(1, in_b);

  CalibrationSession session(CalibrationOptions{
      .collect_histogram = true,
      .num_bins = 16,
      .histogram_tensor_names = {{"input_a"}},
  });

  ASSERT_OK_AND_ASSIGN(const auto stats, session.Invoke(interpreter_.get(), 0));
  // Both tensors still get per-invoke min/max.
  EXPECT_EQ(stats.count("input_a"), 1);
  EXPECT_EQ(stats.count("input_b"), 1);

  // Only input_a gets a histogram.
  const auto hists = session.GetHistograms();
  EXPECT_EQ(hists.count("input_a"), 1);
  EXPECT_EQ(hists.count("input_b"), 0);
}

TEST_F(TflCalibrationUtilsTest,
       CalibrationSessionWithoutCollectHistogramReturnsNoHistograms) {
  AddTensor("input", kTfLiteFloat32, {1, 2});
  interpreter_->SetInputs({0});
  AddNoOpNode({0});
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  const float in[] = {-1.0f, 1.0f};
  PopulateTensor(0, in);

  CalibrationSession session;
  ASSERT_OK_AND_ASSIGN(const auto stats, session.Invoke(interpreter_.get(), 0));
  EXPECT_EQ(stats.count("input"), 1);
  EXPECT_THAT(session.GetHistograms(), IsEmpty());
}

TEST_F(TflCalibrationUtilsTest, CalibrationSessionCustomValidRange) {
  AddTensor("input", kTfLiteFloat32, {1, 6});
  interpreter_->SetInputs({0});
  AddNoOpNode({0});
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  // Bounds are exclusive: -1.0f and 1.0f are filtered out.
  const float in[] = {-5.0f, -1.0f, -0.5f, 0.5f, 1.0f, 5.0f};
  PopulateTensor(0, in);

  CalibrationSession session(CalibrationOptions{
      .valid_min = -1.0f,
      .valid_max = 1.0f,
      .collect_histogram = true,
      .num_bins = 16,
  });
  ASSERT_OK_AND_ASSIGN(const auto stats, session.Invoke(interpreter_.get(), 0));
  EXPECT_FLOAT_EQ(stats.at("input").min, -0.5f);
  EXPECT_FLOAT_EQ(stats.at("input").max, 0.5f);

  const auto hists = session.GetHistograms();
  ASSERT_EQ(hists.count("input"), 1);
  EXPECT_FLOAT_EQ(hists.at("input").min, -0.5f);
  EXPECT_FLOAT_EQ(hists.at("input").max, 0.5f);
  EXPECT_EQ(SumCounts(hists.at("input")), 2);
}

TEST_F(TflCalibrationUtilsTest,
       CalibrationSessionCollectsHistogramsForF16BF16AndInt32) {
  AddTensor("input_f16", kTfLiteFloat16, {1, 4});
  AddTensor("input_bf16", kTfLiteBFloat16, {1, 4});
  AddTensor("input_i32", kTfLiteInt32, {1, 4});
  interpreter_->SetInputs({0, 1, 2});
  AddNoOpNode({0, 1, 2});
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  constexpr float kInf = std::numeric_limits<float>::infinity();
  const Eigen::half in_f16[] = {Eigen::half(-1.5f), Eigen::half(0.0f),
                                Eigen::half(4.0f), Eigen::half(kInf)};
  const Eigen::bfloat16 in_bf16[] = {
      Eigen::bfloat16(-2.0f), Eigen::bfloat16(1.0f), Eigen::bfloat16(8.0f),
      Eigen::bfloat16(-kInf)};
  const int32_t in_i32[] = {-10, 0, 10, 20};
  PopulateTensor(0, in_f16);
  PopulateTensor(1, in_bf16);
  PopulateTensor(2, in_i32);

  CalibrationSession session(CalibrationOptions{
      .collect_histogram = true,
      .num_bins = 16,
  });
  ASSERT_OK(session.Invoke(interpreter_.get(), 0));

  const auto hists = session.GetHistograms();
  ASSERT_EQ(hists.size(), 3);
  // Infinities are excluded from float histograms.
  EXPECT_FLOAT_EQ(hists.at("input_f16").min, -1.5f);
  EXPECT_FLOAT_EQ(hists.at("input_f16").max, 4.0f);
  EXPECT_EQ(SumCounts(hists.at("input_f16")), 3);
  EXPECT_FLOAT_EQ(hists.at("input_bf16").min, -2.0f);
  EXPECT_FLOAT_EQ(hists.at("input_bf16").max, 8.0f);
  EXPECT_EQ(SumCounts(hists.at("input_bf16")), 3);
  EXPECT_FLOAT_EQ(hists.at("input_i32").min, -10.0f);
  EXPECT_FLOAT_EQ(hists.at("input_i32").max, 20.0f);
  EXPECT_EQ(SumCounts(hists.at("input_i32")), 4);
}

TEST_F(TflCalibrationUtilsTest,
       CalibrationSessionMasksOutZeroMaskTokenPositionsFloatAndInt32) {
  AddTensor("encoder_1x4_embeddings:0", kTfLiteFloat32, {1, 4, 2});
  AddTensor("encoder_1x4_mask:0", kTfLiteFloat32, {1, 4});
  interpreter_->SetInputs({0, 1});
  AddNoOpNode({0, 1});
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

  // Only the first 2 token positions (4 elements) are active (mask == 1);
  // the last 2 token positions (mask == 0) contain large padding outliers
  // (-100.0f, 100.0f) that should be excluded from stats and histograms.
  const float emb[] = {
      -1.0f, 0.5f, -0.5f, 2.0f, -100.0f, 100.0f, -100.0f, 100.0f,
  };
  const float mask[] = {1.0f, 1.0f, 0.0f, 0.0f};
  PopulateTensor(0, emb);
  PopulateTensor(1, mask);

  CalibrationSession session(CalibrationOptions{
      .collect_histogram = true,
      .num_bins = 16,
  });
  ASSERT_OK_AND_ASSIGN(const auto stats, session.Invoke(interpreter_.get(), 0));
  EXPECT_FLOAT_EQ(stats.at("encoder_1x4_embeddings:0").min, -1.0f);
  EXPECT_FLOAT_EQ(stats.at("encoder_1x4_embeddings:0").max, 2.0f);

  const auto hists = session.GetHistograms();
  ASSERT_EQ(hists.count("encoder_1x4_embeddings:0"), 1);
  EXPECT_FLOAT_EQ(hists.at("encoder_1x4_embeddings:0").min, -1.0f);
  EXPECT_FLOAT_EQ(hists.at("encoder_1x4_embeddings:0").max, 2.0f);
  EXPECT_EQ(SumCounts(hists.at("encoder_1x4_embeddings:0")), 4);

  // Also verify int32 mask input tensor.
  interpreter_ = std::make_unique<tflite::Interpreter>();
  AddTensor("encoder_1x4_embeddings:0", kTfLiteFloat32, {1, 4, 2});
  AddTensor("encoder_1x4_mask:0", kTfLiteInt32, {1, 4});
  interpreter_->SetInputs({0, 1});
  AddNoOpNode({0, 1});
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  const int32_t mask_i32[] = {1, 1, 0, 0};
  PopulateTensor(0, emb);
  PopulateTensor(1, mask_i32);

  session.Reset();
  ASSERT_OK_AND_ASSIGN(const auto stats_i32,
                       session.Invoke(interpreter_.get(), 0));
  EXPECT_FLOAT_EQ(stats_i32.at("encoder_1x4_embeddings:0").min, -1.0f);
  EXPECT_FLOAT_EQ(stats_i32.at("encoder_1x4_embeddings:0").max, 2.0f);
}

}  // namespace
}  // namespace odml
