// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ml_drift_delegate/delegate/delegate_kernel_litert.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_tensor.h"  // from @ml_drift
#include "ml_drift/common/task/profiling_info.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_common.h"
#include "ml_drift_delegate/delegate/gpu_backend.h"
#include "weight_loader/external_weight_loader_litert.h"
#include "tflite/c/c_api_types.h"
#include "tflite/c/common.h"

namespace litert::ml_drift {
namespace {

using ::testing::HasSubstr;

class DelegateKernelLiteRtTest : public ::testing::Test {
 protected:
  void SetUp() override {
    tensors_.resize(6);
    for (size_t i = 0; i < tensors_.size(); ++i) {
      tensors_[i] = {};
      tensors_[i].type = kTfLiteFloat32;
      tensors_[i].allocation_type = kTfLiteArenaRw;
    }
    context_.tensors = tensors_.data();
    context_.tensors_size = tensors_.size();
  }

  TfLiteContext context_ = {};
  std::vector<TfLiteTensor> tensors_;
};

TEST_F(DelegateKernelLiteRtTest,
       GetEffectiveTensorNoQuantizedTensorsReturnsOriginal) {
  DelegateKernelLiteRt kernel(nullptr);
  EXPECT_FALSE(kernel.HasQuantizedTensors());
  EXPECT_TRUE(kernel.quant_conversion_map().empty());

  for (int i = 0; i < context_.tensors_size; ++i) {
    EXPECT_EQ(kernel.GetEffectiveTensor(&context_, i), &context_.tensors[i]);
  }
}

TEST_F(DelegateKernelLiteRtTest,
       GetEffectiveTensorWithQuantizedMappingResolvesTensors) {
  DelegateKernelLiteRt kernel(nullptr);
  // Map synthetic temporary tensor IDs (4, 5) to signature tensor IDs (0, 1),
  // and vice-versa for bidirectional lookup.
  absl::flat_hash_map<int, int> quant_map = {
      {4, 0},
      {0, 4},
      {5, 1},
      {1, 5},
  };
  kernel.set_quant_conversion_map(std::move(quant_map));
  EXPECT_TRUE(kernel.HasQuantizedTensors());

  // Synthetic tensors 4 and 5 resolve to signature tensors 0 and 1.
  EXPECT_EQ(kernel.GetEffectiveTensor(&context_, 4), &context_.tensors[0]);
  EXPECT_EQ(kernel.GetEffectiveTensor(&context_, 5), &context_.tensors[1]);

  // Signature tensors 0 and 1 resolve to synthetic tensors 4 and 5.
  EXPECT_EQ(kernel.GetEffectiveTensor(&context_, 0), &context_.tensors[4]);
  EXPECT_EQ(kernel.GetEffectiveTensor(&context_, 1), &context_.tensors[5]);

  // Unmapped tensors (e.g. index 2, 3) fall back to returning themselves.
  EXPECT_EQ(kernel.GetEffectiveTensor(&context_, 2), &context_.tensors[2]);
  EXPECT_EQ(kernel.GetEffectiveTensor(&context_, 3), &context_.tensors[3]);
}

class FakeGpuInferenceContext : public GpuInferenceContext {
 public:
  absl::StatusOr<::ml_drift::GpuSpatialTensor*> GetSpatialTensor(
      ::ml_drift::ValueId) override {
    return nullptr;
  }
  absl::Status BindSpatialTensor(::ml_drift::ValueId,
                                 ::ml_drift::GpuSpatialTensor*) override {
    return absl::OkStatus();
  }
  absl::Status UploadWeightsOnWeb(
      weight_loader::WeightLoader*, const ::ml_drift::GpuModel&,
      const absl::flat_hash_map<::ml_drift::ValueId, ::ml_drift::ValueId>&,
      const absl::flat_hash_map<::ml_drift::ValueId, uint32_t>&,
      std::vector<::ml_drift::WeightsManager::UploadWeightsInfo>&) override {
    return absl::OkStatus();
  }
  absl::Status WriteDataToWeightTensor(::ml_drift::ValueId,
                                       absl::Span<const uint8_t>) override {
    return absl::OkStatus();
  }
  absl::Status ReadWeightTensorToDescriptor(
      ::ml_drift::ValueId, ::ml_drift::TensorDescriptor&) override {
    return absl::OkStatus();
  }
  absl::Status Dispatch() override { return absl::OkStatus(); }
  absl::StatusOr<GpuEventHandle> GetPreDispatchEvent() override {
    return absl::NotFoundError("No pre-dispatch event");
  }
  absl::StatusOr<GpuEventHandle> GetPostDispatchEvent(
      bool is_async_execution_mode) override {
    if (is_async_execution_mode) {
      return GpuEventHandle(reinterpret_cast<void*>(0x1234));
    }
    return absl::NotFoundError("Sync mode has no post-dispatch event");
  }
  absl::Status WaitForEventsCompleted(absl::Span<GpuEventHandle>,
                                      bool) override {
    return absl::OkStatus();
  }
  absl::Status PreConvert(bool input) override { return absl::OkStatus(); }
  absl::Status PostConvert(bool input) override { return absl::OkStatus(); }
  absl::Status Profile(::ml_drift::ProfilingInfo& profiling_info) override {
    return absl::OkStatus();
  }
  absl::StatusOr<size_t> GetSizeOfMemoryAllocatedForIntermediateTensors()
      const override {
    return 0;
  }
  absl::StatusOr<size_t> GetSizeOfMemoryAllocatedForConstantTensors()
      const override {
    return 0;
  }
  absl::StatusOr<size_t> GetSizeOfMemoryAllocatedForExternalTensors()
      const override {
    return 0;
  }
  absl::Status ReportMemoryBenchmarkIfEnabled(
      const ::ml_drift::CreateGpuModelInfo& create_info) override {
    return absl::OkStatus();
  }
  absl::Status SetCommandBufferHint(
      int num_nodes_per_command_encoder) override {
    return absl::OkStatus();
  }
};

class TestableDelegateKernelLiteRt : public DelegateKernelLiteRt {
 public:
  explicit TestableDelegateKernelLiteRt(
      const LiteRtRuntimeContext* runtime_context)
      : DelegateKernelLiteRt(runtime_context) {}

  void SetGpuInferenceContext(std::unique_ptr<GpuInferenceContext> ctx) {
    ctx_ = std::move(ctx);
  }

  void SetBufferContext(LiteRtExternalLiteRtBufferContext buffer_context) {
    buffer_context_ = buffer_context;
  }

  void SetOutputIndices(std::vector<int64_t> output_indices) {
    output_indices_ = std::move(output_indices);
  }

  void SetInputIndices(std::vector<int64_t> input_indices) {
    input_indices_ = std::move(input_indices);
    input_ids_.resize(input_indices_.size());
  }

  void SetOutputIds(std::vector<::ml_drift::ValueId> ids) {
    output_ids_ = std::move(ids);
  }

  void SetExternalTensorIds(absl::flat_hash_set<::ml_drift::ValueId> ids) {
    external_tensor_ids_ = std::move(ids);
  }
};

TEST_F(DelegateKernelLiteRtTest,
       UploadIntermediateCpuTensorsResolvesEffectiveTensor) {
  TestableDelegateKernelLiteRt kernel(nullptr);
  kernel.SetInputIndices({5});
  kernel.set_quant_conversion_map({{5, 0}, {0, 5}});
  kernel.SetExternalTensorIds({0});

  // Tensors are arena allocated (not kTfLiteCustom), so after resolving the
  // effective tensor it skips upload and returns OK.
  EXPECT_TRUE(kernel.UploadIntermediateCpuTensorsToGpuMemory(&context_).ok());
}

TEST_F(DelegateKernelLiteRtTest,
       DownloadGpuMemoryToIntermediateCpuTensorsResolvesEffectiveTensor) {
  TestableDelegateKernelLiteRt kernel(nullptr);
  kernel.SetOutputIndices({4});
  kernel.SetOutputIds({1});
  kernel.set_quant_conversion_map({{4, 1}, {1, 4}});
  kernel.SetExternalTensorIds({1});

  // Tensors are arena allocated (not kTfLiteCustom), so after resolving the
  // effective tensor it skips download and returns OK.
  EXPECT_TRUE(kernel.DownloadGpuMemoryToIntermediateCpuTensors(&context_).ok());
}

TEST_F(DelegateKernelLiteRtTest,
       HandleOutputEventsResolvesEffectiveTensorInAsyncMode) {
  LiteRtRuntimeContext runtime_context = {};
  runtime_context.external_litert_buffer_context_get_environment =
      [](LiteRtExternalLiteRtBufferContext,
         LiteRtEnvironment* env) -> LiteRtStatus {
    *env = reinterpret_cast<LiteRtEnvironment>(0x5678);
    return kLiteRtStatusOk;
  };
  runtime_context.get_external_litert_buffer_context_tensor_buffer =
      [](LiteRtExternalLiteRtBufferContext, const TfLiteTensor*,
         LiteRtTensorBuffer*) -> LiteRtStatus {
    return kLiteRtStatusErrorNotFound;
  };

  TestableDelegateKernelLiteRt kernel(&runtime_context);
  kernel.SetGpuInferenceContext(std::make_unique<FakeGpuInferenceContext>());
  kernel.SetBufferContext(
      reinterpret_cast<LiteRtExternalLiteRtBufferContext>(0x9abc));
  kernel.SetOutputIndices({4});
  kernel.set_quant_conversion_map({{4, 1}, {1, 4}});

  // In async mode, HandleOutputEvents resolves the effective tensor for output
  // index 4 (mapping to signature tensor 1) and queries the buffer context.
  // Since the tensor buffer is not registered, it returns an error verifying
  // that effective tensor resolution and lookup occurred.
  auto status =
      kernel.HandleOutputEvents(&context_, /*is_async_execution_mode=*/true);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(
      status.message(),
      HasSubstr("TensorBuffer is not registered for output#0 tensor T#4."));
}

TEST_F(DelegateKernelLiteRtTest, HandleInputEventsResolvesEffectiveTensor) {
  LiteRtRuntimeContext runtime_context = {};
  runtime_context.get_external_litert_buffer_context_tensor_buffer =
      [](LiteRtExternalLiteRtBufferContext, const TfLiteTensor*,
         LiteRtTensorBuffer*) -> LiteRtStatus {
    return kLiteRtStatusErrorNotFound;
  };

  TestableDelegateKernelLiteRt kernel(&runtime_context);
  kernel.SetBufferContext(
      reinterpret_cast<LiteRtExternalLiteRtBufferContext>(0x9abc));
  kernel.SetInputIndices({5});
  kernel.set_quant_conversion_map({{5, 0}, {0, 5}});

  auto status = kernel.HandleInputEvents(&context_);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(
      status.message(),
      HasSubstr("TensorBuffer is not registered for input#0 tensor T#5."));
}

}  // namespace
}  // namespace litert::ml_drift
