/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

// #include <NvInfer.h>
// #include <NvInferPlugin.h>

// #include <cufft.h>
// #include <cufftXt.h>
// #include <cublas_v2.h>

// #include <array>
// #include <cassert>
// #include <cstring>
// #include <iostream>
// #include <memory>
// #include <string>
// #include <vector>
#include "dft_plugins.h"

namespace trt_dft {

using namespace nvinfer1;
using nvinfer1::plugin::FftPlugin;
using nvinfer1::plugin::IfftPlugin;
// using nvinfer1::plugin::

static const char* DFT_PLUGIN_VERSION{"1"};
static const char* FFT_PLUGIN_NAME{"Fft"};
static const char* IFFT_PLUGIN_NAME{"Ifft"};

// DFT plugins base class.
//
FftPluginBase::FftPluginBase(int direction, int32_t normalized, int32_t onesided, int32_t signal_ndim):
    direction_(direction),
    normalized_(normalized),
    onesided_(onesided),
    signal_ndim_(signal_ndim) {
    // This mimics limitations of ONNX Contrib ops.
    assert(normalized == 0);
    assert(onesided == 1);
    assert(1 <= signal_ndim && signal_ndim <= 3);
}

    // Deserialization ctor.
FftPluginBase::FftPluginBase(void const* data, size_t size) {
    assert(data != nullptr);
    assert(size == getSerializationSize());

    auto p = data;
    direction_ = readBuf<decltype(direction_)>(p);
    normalized_ = readBuf<decltype(normalized_)>(p);
    onesided_ = readBuf<decltype(onesided_)>(p);
    signal_ndim_ = readBuf<decltype(signal_ndim_)>(p);

    assert(reinterpret_cast<char const*>(data) + size == p);
}

IPluginV2DynamicExt* FftPluginBase::clone() const {
    try {
        auto plugin = cloneImpl();
        plugin->setPluginNamespace(ns_.c_str());
        return plugin;
    }
    catch(const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
    return nullptr;
}

AsciiChar const* FftPluginBase::getPluginVersion() const {
    return DFT_PLUGIN_VERSION.c_str();
}

int32_t FftPluginBase::getNbOutputs() const {
    return 1;
}

bool FftPluginBase::supportsFormatCombination(int32_t pos, PluginTensorDesc const* inOut,
                               int32_t nbInputs, int32_t nbOutputs)
                               noexcept {
    assert(0 <= pos && pos < nbInputs + nbOutputs);
    assert(nbInputs == 1);
    assert(nbOutputs == 1);

    bool supported = true;
    supported &= inOut[pos].format == TensorFormat::kLINEAR;
    supported &= inOut[pos].type == DataType::kFLOAT;

    return supported;
}

int32_t FftPluginBase::initialize() noexcept {
    return 0;
}

void FftPluginBase::terminate() noexcept {
}

size_t FftPluginBase::getWorkspaceSize(PluginTensorDesc const* inputs, int32_t nbInputs,
                        PluginTensorDesc const* outputs, int32_t nbOutputs)
                        const noexcept {
    // Make sure inputs/outputs are the same as were provided in configure earlier.
    assert(nbInputs == 1);
    assert(nbOutputs == 1);
    // TODO(akamenev): next 2 asserts fail because PluginTensorDesc.scale
    // are not the same between configure and getWorkspaceSize calls.
    // assert(std::memcmp(inputs, &in_desc_.desc, sizeof(PluginTensorDesc)) == 0);
    // assert(std::memcmp(inputs, &out_desc_.desc, sizeof(PluginTensorDesc)) == 0);

    size_t res = 0;
    for (auto n : ws_size_)
        res += n;
    return res;
}

void FftPluginBase::configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs,
                     DynamicPluginTensorDesc const* out, int32_t nbOutputs)
                     noexcept {
    assert(nbInputs == 1);
    assert(nbOutputs == 1);
    const auto& in0 = in[0];
    const auto& out0 = out[0];
    assert(in0.desc.type == DataType::kFLOAT);
    assert(in0.desc.format == TensorFormat::kLINEAR);
    assert(out0.desc.type == DataType::kFLOAT);
    assert(out0.desc.format == TensorFormat::kLINEAR);

    in_desc_ = in0;
    out_desc_ = out0;

    // TODO(akamenev): max/min dims are not yet supported, so check that.
    assert(in0.desc.dims.nbDims == in0.min.nbDims);
    assert(in0.desc.dims.nbDims == in0.max.nbDims);
    auto in0_dims_b = std::begin(in0.desc.dims.d);
    auto in0_dims_e = in0_dims_b + in0.desc.dims.nbDims;
    assert(std::equal(in0_dims_b, in0_dims_e, std::begin(in0.min.d)));
    assert(std::equal(in0_dims_b, in0_dims_e, std::begin(in0.max.d)));

    // TODO(akamenev): according to TRT docs:
    // https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#ipluginv2
    // configurePlugin should not be used to allocate resources, however,
    // to create cuFFT plan we need to know dimensions of the input/output,
    // and this information is not yet available in initialize() which is called
    // before configurePlugin.

    handle_ = cufft_ptr(createCufftHandle());

    // Disable cuFFT workspace auto-allocation as we'll be using
    // TensorRT-allocated workspace.
    auto err = cufftSetAutoAllocation(*handle_, 0);
    assert(err == CUFFT_SUCCESS);

    // Create cuFFT plan.
    auto [batch_size, dft_dims] = splitSignalDims();
    auto in_out_types = getInOutTypes();
    err = cufftXtMakePlanMany(*handle_, signal_ndim_, dft_dims.data(),
                              /*inembed*/nullptr, 1, 0, std::get<0>(in_out_types),
                              /*onembed*/nullptr, 1, 0, std::get<1>(in_out_types),
                              /*batch*/batch_size,
                              /*workSize*/ws_size_.data(),
                              /*executiontype*/CUDA_C_32F);
    assert(err == CUFFT_SUCCESS);
}

int32_t FftPluginBase::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
                void const* const* inputs, void* const* outputs,
                void* workspace, cudaStream_t stream) noexcept override {
    static_assert(direction_ == CUFFT_FORWARD || direction_ == CUFFT_INVERSE);

    auto err = cufftSetStream(*handle_, stream);
    assert(err == CUFFT_SUCCESS);

    // Set work area.
    err = cufftSetWorkArea(*handle_, workspace);
    assert(err == CUFFT_SUCCESS);

    err = cufftXtExec(*handle_,
                      const_cast<void*>(inputs[0]),
                      const_cast<void*>(outputs[0]),
                      direction_);
    assert(err == CUFFT_SUCCESS);

    return 0;
}

size_t FftPluginBase::getSerializationSize() const noexcept {
    // ATTENTION: don't forget to update this method after changing serialize().
    return sizeof(normalized_) + sizeof(onesided_) + sizeof(signal_ndim_);
}

void FftPluginBase::serialize(void* buffer) const noexcept {
    // ATTENTION: when changing this method, don't forget to update
    // getSerializationSize() accordingly.
    assert(buffer != nullptr);

    auto p = buffer;
    writeBuf(normalized_, p);
    writeBuf(onesided_, p);
    writeBuf(signal_ndim_, p);

    auto expected_size = getSerializationSize();
    assert(reinterpret_cast<char const*>(buffer) + expected_size == p);
}

void FftPluginBase::destroy() noexcept {
    delete this;
}

void FftPluginBase::setPluginNamespace(AsciiChar const* pluginNamespace) noexcept {
    ns_ = pluginNamespace;
}

AsciiChar const* FftPluginBase::getPluginNamespace() const noexcept {
    return ns_.c_str();
}

DataType FftPluginBase::getOutputDataType(int32_t index,
                           DataType const* inputTypes,
                           int32_t nbInputs) const noexcept {
    assert(nbInputs == 1);
    assert(index == 0);
    assert(inputTypes[0] == DataType::kFLOAT);

    return inputTypes[0];
}

// Splits total signal dims into batch size and DFT signal dims.
std::pair<int32_t, std::array<long long, 3>> FftPluginBase::splitSignalDims() {
    auto dims = getSignalDims();

    assert(dims.nbDims >= signal_ndim_);

    // cuFFT supports only 1D, 2D and 3D DFTs.
    std::array<long long, 3> dft_dims;

    // subtract 1 since the last dimension will be real/imag values
    int32_t dim_start = dims.nbDims - signal_ndim_ - 1;
    for (int32_t i = 0; i < signal_ndim_; i++)
        dft_dims[i] = dims.d[dim_start + i];
    // Fold other dimensions into a single batch dim.
    int32_t batch_size = 1;
    for (int32_t i = 0; i < dim_start; i++)
        batch_size *= dims.d[i];

    return {batch_size, dft_dims};
}

 
// cuFFT helpers.
//
cufftHandle* FftPluginBase::createCufftHandle() const {
    auto res = new cufftHandle{};
    auto err = cufftCreate(res);
    assert(err == CUFFT_SUCCESS);
    return res;
}

// cuBLAS helpers.
//
cublasHandle_t* FftPluginBase::createCublasHandle() const {
    auto res = new cublasHandle_t{};
    auto err = cublasCreate(res);
    assert(err == CUBLAS_STATUS_SUCCESS);
    return res;
}




FftPlugin::FftPlugin(int32_t normalized, int32_t onesided, int32_t signal_ndim):
    FftPluginBase(CUFFT_FORWARD, normalized, onesided, signal_ndim) {
}

// Deserialization ctor.
FftPlugin::FftPlugin(void const* data, size_t size):
    FftPluginBase(data, size) {
}

AsciiChar const* FftPlugin::getPluginType() const noexcept override {
    return FFT_PLUGIN_NAME;
}

DimsExprs FftPlugin::getOutputDimensions(int32_t outputIndex,
                              DimsExprs const* inputs, int32_t nbInputs,
                              IExprBuilder& exprBuilder) noexcept override {
    assert(outputIndex == 0);
    assert(nbInputs == 1);

    DimsExprs output(inputs[0]);
    // RFFT output is complex, so add a dimension for complex number representation.
    assert(output.nbDims < Dims::MAX_DIMS);
    return output;
}

FftPluginBase* FftPlugin::cloneImpl() const {
    return new FftPlugin(normalized_, onesided_, signal_ndim_);
}

std::pair<cudaDataType, cudaDataType> FftPlugin::getInOutTypes() const {
    return {CUDA_C_32F, CUDA_C_32F};
}

Dims FftPlugin::getSignalDims() const { return in_desc_.desc.dims; }


IfftPlugin::IfftPlugin(int32_t normalized, int32_t onesided, int32_t signal_ndim):
    FftPluginBase(CUFFT_INVERSE,normalized, onesided, signal_ndim) {
}

    // Deserialization ctor.
IfftPlugin:IfftPlugin(void const* data, size_t size):
    FftPluginBase(data, size) {
}

AsciiChar const* IfftPlugin::getPluginType() const {
    return IFFT_PLUGIN_NAME;
}

DimsExprs IfftPlugin::getOutputDimensions(int32_t outputIndex,
                              DimsExprs const* inputs, int32_t nbInputs,
                              IExprBuilder& exprBuilder) noexcept override {
    assert(outputIndex == 0);
    assert(nbInputs == 1);

    DimsExprs output(inputs[0]);
    // IRFFT input is complex, output is real, so remove the last dimension
    // used for complex numbers representation.
    assert(output.nbDims > 1);
    // output.nbDims -= 1;
    // Since the input is one-sided, Hermitian signal, the real-valued
    // output will have the last dimension of (N - 1) * 2.
    // output.d[output.nbDims - 1] = exprBuilder.operation(
    //     DimensionOperation::kPROD,
    //     *exprBuilder.operation(
    //         DimensionOperation::kSUB,
    //         *output.d[output.nbDims - 1],
    //         *exprBuilder.constant(1)),
    //     *exprBuilder.constant(2));
    return output;
}

void IfftPlugin::configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs,
                     DynamicPluginTensorDesc const* out, int32_t nbOutputs)
                     noexcept override {
    Base::configurePlugin(in, nbInputs, out, nbOutputs);
    cublas_ = cublas_ptr(createCublasHandle());
}

int32_t IfftPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
                void const* const* inputs, void* const* outputs,
                void* workspace, cudaStream_t stream) {
    int32_t err{0};
    err = Base::enqueue(inputDesc, outputDesc, inputs, outputs,
                        workspace, stream);
    if (err != 0)
        return err;

    // TODO: implement normalization schemes

    err = cublasSetStream(*cublas_, stream);
    assert(err == CUBLAS_STATUS_SUCCESS);

    // Scale the output to mimic ONNX Contrib IRFFT behavior
    // aka "backward" normalization mode in PyTorch fft.
    auto [batch_size, dft_dims] = splitSignalDims();
    float total_dft_size = 1.0f;
    for (int i = 0; i < signal_ndim_; i++)
        total_dft_size *= dft_dims[i];

    float scale = 1.0f / total_dft_size;
    float complex_scale = make_float2(scale, 0.0f);
    err = cublasScalEx(*cublas_, batch_size * total_dft_size,
                       &complex_scale, CUDA_C_32F,
                       outputs[0], CUDA_C_32F, 1,
                       CUDA_C_32F);
    assert(err == CUBLAS_STATUS_SUCCESS);

    return 0;
}

// using Base = FftPluginBase<CUFFT_INVERSE>;

FftPluginBase* cloneImpl() const {
    return new IfftPlugin(normalized_, onesided_, signal_ndim_);
}

std::pair<cudaDataType, cudaDataType> getInOutTypes() const {
    return {CUDA_C_32F, CUDA_C_32F};
}

Dims getSignalDims() const { return out_desc_.desc.dims; }


// Plugin creators.
//

// FFT plugin creator
FftPluginCreator::FftPluginCreator() {
    attrs_.emplace_back(PluginField{"normalized", nullptr, PluginFieldType::kINT32, 1});
    attrs_.emplace_back(PluginField{"onesided", nullptr, PluginFieldType::kINT32, 1});
    attrs_.emplace_back(PluginField{"signal_ndim", nullptr, PluginFieldType::kINT32, 1});

    field_names_.nbFields = attrs_.size();
    field_names_.fields = attrs_.data();
}

AsciiChar const* FftPluginCreator::getPluginName() const {
    return FFT_PLUGIN_NAME;
}

AsciiChar const* FftPluginCreator::getPluginVersion() const {
    return DFT_PLUGIN_VERSION.c_str();
}

PluginFieldCollection const* FftPluginCreator::getFieldNames() {
    return &field_names_;
}

IPluginV2* FftPluginCreator::createPlugin(AsciiChar const* name, PluginFieldCollection const* fc) {
    try {
        const PluginField* fields = fc->fields;
        assert(fc->nbFields == 3);
        assert(fields[0].type == PluginFieldType::kINT32);
        assert(!strcmp(fields[0].name, "normalized"));
        assert(fields[1].type == PluginFieldType::kINT32);
        assert(!strcmp(fields[1].name, "onesided"));
        assert(fields[2].type == PluginFieldType::kINT32);
        assert(!strcmp(fields[2].name, "signal_ndim"));

        int32_t normalized = *(static_cast<const int32_t*>(fields[0].data));
        int32_t onesided = *(static_cast<const int32_t*>(fields[1].data));
        int32_t signal_ndim = *(static_cast<const int32_t*>(fields[2].data));

        return new FftPlugin(normalized, onesided, signal_ndim);
    }
    catch(const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
    return nullptr;
}

IPluginV2* FftPluginCreator::deserializePlugin(AsciiChar const* name, void const* serialData, size_t serialLength) {
    try {
        return new FftPlugin(serialData, serialLength);
    }
    catch(const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }

    return nullptr;
}

void FftPluginCreator::setPluginNamespace(AsciiChar const* pluginNamespace) {
    ns_ = pluginNamespace;
}

AsciiChar const* FftPluginCreator::getPluginNamespace() const {
    return ns_.c_str();
}

//IFFT plugin factory

IfftPluginCreator::IfftPluginCreator() {
    attrs_.emplace_back(PluginField{"normalized", nullptr, PluginFieldType::kINT32, 1});
    attrs_.emplace_back(PluginField{"onesided", nullptr, PluginFieldType::kINT32, 1});
    attrs_.emplace_back(PluginField{"signal_ndim", nullptr, PluginFieldType::kINT32, 1});

    field_names_.nbFields = attrs_.size();
    field_names_.fields = attrs_.data();
}

AsciiChar const* IfftPluginCreator::getPluginName() const {
    return IFFT_PLUGIN_NAME;
}

AsciiChar const* IfftPluginCreator::getPluginVersion() const {
    return DFT_PLUGIN_VERSION.c_str();
}

PluginFieldCollection const* IfftPluginCreator::getFieldNames() {
    return &field_names_;
}

IPluginV2* IfftPluginCreator::createPlugin(AsciiChar const* name, PluginFieldCollection const* fc) {
    try {
        const PluginField* fields = fc->fields;
        assert(fc->nbFields == 3);
        assert(fields[0].type == PluginFieldType::kINT32);
        assert(!strcmp(fields[0].name, "normalized"));
        assert(fields[1].type == PluginFieldType::kINT32);
        assert(!strcmp(fields[1].name, "onesided"));
        assert(fields[2].type == PluginFieldType::kINT32);
        assert(!strcmp(fields[2].name, "signal_ndim"));

        int32_t normalized = *(static_cast<const int32_t*>(fields[0].data));
        int32_t onesided = *(static_cast<const int32_t*>(fields[1].data));
        int32_t signal_ndim = *(static_cast<const int32_t*>(fields[2].data));

        return new IfftPlugin(normalized, onesided, signal_ndim);
    }
    catch(const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
    return nullptr;
}

IPluginV2* IfftPluginCreator::deserializePlugin(AsciiChar const* name, void const* serialData, size_t serialLength) {
    try {
        return new IfftPlugin(serialData, serialLength);
    }
    catch(const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }

    return nullptr;
}

void IfftPluginCreator::setPluginNamespace(AsciiChar const* pluginNamespace) {
    ns_ = pluginNamespace;
}

AsciiChar const* IfftPluginCreator::getPluginNamespace() const {
    return ns_.c_str();
}

 // namespace trt_dft
