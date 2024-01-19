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
#ifndef DFT_PLUGINS_H
#define DFT_PLUGINS_H
#include <NvInfer.h>
#include <NvInferPlugin.h>

#include <cufft.h>
#include <cufftXt.h>
#include <cublas_v2.h>

#include <array>
#include <cassert>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>



namespace nvinfer1 {
    namespace plugin {


    // DFT plugins base class.

    class FftPluginBase: public IPluginV2DynamicExt {
     public:
        FftPluginBase(int direction, int32_t normalized, int32_t onesided, int32_t signal_ndim);
        // Deserialization ctor.
        FftPluginBase(void const* data, size_t size);

        IPluginV2DynamicExt* clone() const override;

        AsciiChar const* getPluginVersion() const override;

        int32_t getNbOutputs() const override;

        bool supportsFormatCombination(int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) override;

        int32_t initialize() override;

        void terminate() override;

        size_t getWorkspaceSize(PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs, int32_t nbOutputs) const override;

        void configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) override;

        int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) override ;

        size_t getSerializationSize() const override ;

        void serialize(void* buffer) const override ;

        void destroy() override ;

        void setPluginNamespace(AsciiChar const* pluginNamespace) override ;

        AsciiChar const* getPluginNamespace() const override ;

        DataType getOutputDataType(int32_t index, DataType const* inputTypes, int32_t nbInputs) const override;

     protected:
        virtual FftPluginBase* cloneImpl() const = 0;

        virtual std::pair<cudaDataType, cudaDataType> getInOutTypes() const = 0;

        virtual Dims getSignalDims() const = 0;

        // Splits total signal dims into batch size and DFT signal dims.
        std::pair<int32_t, std::array<long long, 3>> splitSignalDims() ;
     protected:
        // cuFFT helpers.
        //
        cufftHandle* createCufftHandle() const ;
        
        struct cufftHandleDeleter {
            void operator()(cufftHandle* handle) const {
                assert(handle != nullptr);
                auto err = cufftDestroy(*handle);
                delete handle;
                assert(err == CUFFT_SUCCESS);
            }
        };

        using cufft_ptr = std::unique_ptr<cufftHandle, cufftHandleDeleter>;

        // cuBLAS helpers.
        //
        cublasHandle_t* createCublasHandle() const;
        struct cublasHandleDeleter {
            void operator()(cublasHandle_t* handle) const {
                assert(handle != nullptr);
                auto err = cublasDestroy(*handle);
                delete handle;
                assert(err == CUBLAS_STATUS_SUCCESS);
            }
        };

        using cublas_ptr = std::unique_ptr<cublasHandle_t, cublasHandleDeleter>;

     public:
        // Helper serialization functions.
        //
        template<typename T>
        void writeBuf(const T& val, void*& buffer) const {
            auto size = sizeof(val);
            std::memcpy(buffer, &val, size);
            auto& b = reinterpret_cast<char*&>(buffer);
            b += size;
        }

        template<typename T>
        T readBuf(void const*& buffer) const {
            T val{};
            auto size = sizeof(val);
            std::memcpy(&val, buffer, size);
            auto& b = reinterpret_cast<char const*&>(buffer);
            b += size;
            return val;
        }

     protected:
        int direction_;
        int32_t normalized_{0};
        int32_t onesided_{0};
        int32_t signal_ndim_{0};
        std::string ns_;

        DynamicPluginTensorDesc in_desc_{};
        DynamicPluginTensorDesc out_desc_{};

        // cuFFT data.
        cufft_ptr handle_;

        // cuFFT workspace size.
        // TODO(akamenev): assuming single GPU for now.
        std::vector<size_t> ws_size_{0};
    };


    class FftPlugin: public FftPluginBase {
     public:
        FftPlugin(int32_t normalized, int32_t onesided, int32_t signal_ndim);
        // Deserialization ctor.
        FftPlugin(void const* data, size_t size);

        AsciiChar const* getPluginType() const override ;

        DimsExprs getOutputDimensions(int32_t outputIndex,
                                      DimsExprs const* inputs, int32_t nbInputs,
                                      IExprBuilder& exprBuilder) override ;

     public:
        static constexpr char name[]{"Rfft"};

     protected:
        FftPluginBase* cloneImpl() const override ;
        std::pair<cudaDataType, cudaDataType> getInOutTypes() const override ;

        Dims getSignalDims() const override { return in_desc_.desc.dims; }
    };


    class IfftPlugin: public FftPluginBase<CUFFT_INVERSE> {
     public:
        IfftPlugin(int32_t normalized, int32_t onesided, int32_t signal_ndim);

        // Deserialization ctor.
        IfftPlugin(void const* data, size_t size);

        AsciiChar const* getPluginType() const noexcept override ;

        DimsExprs getOutputDimensions(int32_t outputIndex,
                                      DimsExprs const* inputs, int32_t nbInputs,
                                      IExprBuilder& exprBuilder) noexcept override;

        void configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs,
                             DynamicPluginTensorDesc const* out, int32_t nbOutputs)
                             noexcept override;

        int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
                        void const* const* inputs, void* const* outputs,
                        void* workspace, cudaStream_t stream) noexcept override ;

     public:
        static constexpr char name[]{"Irfft"};

     protected:
        using Base = FftPluginBase<CUFFT_INVERSE>;

        FftPluginBase* cloneImpl() const noexcept override;

        std::pair<cudaDataType, cudaDataType> getInOutTypes() const noexcept override;

        Dims getSignalDims() const noexcept override { return out_desc_.desc.dims; }

     private:
        cublas_ptr cublas_;
    };


    // Plugin creators.
    //
    class FftPluginCreator: public IPluginCreator {
     public:
        FftPluginCreator();

        AsciiChar const* getPluginName() const noexcept override;

        AsciiChar const* getPluginVersion() const noexcept override;

        PluginFieldCollection const* getFieldNames() noexcept override;

        IPluginV2* createPlugin(AsciiChar const* name, PluginFieldCollection const* fc)
                                noexcept override;

        IPluginV2* deserializePlugin(AsciiChar const* name,
                                     void const* serialData,
                                     size_t serialLength) noexcept;

        void setPluginNamespace(AsciiChar const* pluginNamespace) noexcept override;

        AsciiChar const* getPluginNamespace() const noexcept override;

     private:
        std::vector<PluginField> attrs_;
        PluginFieldCollection field_names_{};
        std::string ns_;
    };
    
    class IfftPluginCreator: public IPluginCreator {
     public:
        IfftPluginCreator();

        AsciiChar const* getPluginName() const override;

        AsciiChar const* getPluginVersion() const override;

        PluginFieldCollection const* getFieldNames() override;

        IPluginV2* createPlugin(AsciiChar const* name, PluginFieldCollection const* fc) override;

        IPluginV2* deserializePlugin(AsciiChar const* name,
                                     void const* serialData,
                                     size_t serialLength);

        void setPluginNamespace(AsciiChar const* pluginNamespace) override;

        AsciiChar const* getPluginNamespace() const override;

     private:
        std::vector<PluginField> attrs_;
        PluginFieldCollection field_names_{};
        std::string ns_;
    };

    }
    

}



#endif