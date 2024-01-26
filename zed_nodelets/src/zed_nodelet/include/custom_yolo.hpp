#ifndef _CUSTOM_YOLO_H
#define _CUSTOM_YOLO_H

#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "cuda_utils.h"
#include "NvInfer.h"
#include "logging.h"

#if NV_TENSORRT_MAJOR >= 8
#define TRT_NOEXCEPT noexcept
#define TRT_CONST_ENQUEUE const
#else
#define TRT_NOEXCEPT
#define TRT_CONST_ENQUEUE
#endif

#define USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32

using namespace nvinfer1;

namespace Yolo
{
    public:
    static constexpr int CHECK_COUNT = 3;
    static constexpr int MAX_OUTPUT_BBOX_COUNT = 1000;
    static constexpr int CLASS_NUM = 2;
    static constexpr int INPUT_H = 640;  // yolov5's input height and width must be divisible by 32.
    static constexpr int INPUT_W = 640;
    static constexpr int BATCH_SIZE = 1;
    static constexpr float NMS_THRESH = 0.4;
    static constexpr float CONF_THRESH = 0.5;
    static constexpr int LOCATIONS = 4;
    static const int OUTPUT_SIZE = MAX_OUTPUT_BBOX_COUNT * sizeof (Detection) / sizeof (float) + 1; // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
    const char* INPUT_BLOB_NAME = "data";
    const char* OUTPUT_BLOB_NAME = "prob";

    class Model {
        IRuntime* runtime;
        ICudaEngine* engine;
        IExecutionContext* context;
        cudaStream_t stream;
        Logger gLogger;
        void* buffers[2];

        public:
        void Model(std::string engine_path) {
            // deserialize the .engine and run inference
            std::ifstream file(engine_path, std::ios::binary);
            if (!file.good()) {
                throw new std::runtime_error("Failed to read " + engine_path + "!");
            }
            
            char *trtModelStream = nullptr;
            size_t size = 0;
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();

            runtime = createInferRuntime(gLogger);
            assert(runtime != nullptr);
            // trtModelStream is the serialized model text, size is its length, TOOD
            engine = runtime->deserializeCudaEngine(trtModelStream, size);
            assert(engine != nullptr);
            context = engine->createExecutionContext();
            assert(context != nullptr);
            delete[] trtModelStream;
            assert(engine->getNbBindings() == 2);
            // In order to bind the buffers, we need to know the names of the input and output tensors.
            // Note that indices are guaranteed to be less than IEngine::getNbBindings()
            const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
            const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
            assert(inputIndex == 0);
            assert(outputIndex == 1);
            // Create GPU buffers on device
            CUDA_CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof (float)));
            CUDA_CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof (float)));
            // Create stream
            cudaStream_t stream;
            CUDA_CHECK(cudaStreamCreate(&stream));

            assert(BATCH_SIZE == 1); // This sample only support batch 1 for now
        }

        void doInference(float* input, float* output, int batchSize) {
            // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
            CUDA_CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof (float), cudaMemcpyHostToDevice, stream));
            context->enqueue(batchSize, buffers, stream, nullptr);
            CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof (float), cudaMemcpyDeviceToHost, stream));
            cudaStreamSynchronize(stream);
        }

		float* inferImage(sl::Mat &input) {
			static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
			static float prob[BATCH_SIZE * OUTPUT_SIZE];

			cv::Mat left_cv_rgba = slMat2cvMat(left_sl);
			cv::cvtColor(left_cv_rgba, left_cv_rgb, cv::COLOR_BGRA2BGR);
			if (left_cv_rgb.empty()) return;
			cv::Mat pr_img = preprocess_img(left_cv_rgb, INPUT_W, INPUT_H); // letterbox BGR to RGB
			int i = 0;
			int batch = 0;
			for (int row = 0; row < INPUT_H; ++row) {
				uchar* uc_pixel = pr_img.data + row * pr_img.step;
				for (int col = 0; col < INPUT_W; ++col) {
					data[batch * 3 * INPUT_H * INPUT_W + i] = (float) uc_pixel[2] / 255.0;
					data[batch * 3 * INPUT_H * INPUT_W + i + INPUT_H * INPUT_W] = (float) uc_pixel[1] / 255.0;
					data[batch * 3 * INPUT_H * INPUT_W + i + 2 * INPUT_H * INPUT_W] = (float) uc_pixel[0] / 255.0;
					uc_pixel += 3;
					++i;
				}
			}

			doInference(data, prob, BATCH_SIZE);
			return prob;
		}
    }

    std::vector<sl::uint2> cvt(const cv::Rect &bbox_in){
        std::vector<sl::uint2> bbox_out(4);
        bbox_out[0] = sl::uint2(bbox_in.x, bbox_in.y);
        bbox_out[1] = sl::uint2(bbox_in.x + bbox_in.width, bbox_in.y);
        bbox_out[2] = sl::uint2(bbox_in.x + bbox_in.width, bbox_in.y + bbox_in.height);
        bbox_out[3] = sl::uint2(bbox_in.x, bbox_in.y + bbox_in.height);
        return bbox_out;
    }

    struct alignas(float) Detection {
        //center_x center_y w h
        float bbox[LOCATIONS];
        float conf;  // bbox_conf * cls_conf
        float class_id;
    };

    private:
    struct Kernel
    {
        int width;
        int height;
        float anchors[CHECK_COUNT * 2];
    };
}

namespace nvinfer1
{
    class YoloLayerPlugin : public IPluginV2IOExt
    {
    public:
        YoloLayerPlugin(int classCount, int netWidth, int netHeight, int maxOut, const std::vector<Yolo::YoloKernel>& vYoloKernel);
        YoloLayerPlugin(const void* data, size_t length);
        ~YoloLayerPlugin();

        int getNbOutputs() const TRT_NOEXCEPT override
        {
            return 1;
        }

        Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) TRT_NOEXCEPT override;

        int initialize() TRT_NOEXCEPT override;

        virtual void terminate() TRT_NOEXCEPT override {};

        virtual size_t getWorkspaceSize(int maxBatchSize) const TRT_NOEXCEPT override { return 0; }

        virtual int enqueue(int batchSize, const void* const* inputs, void*TRT_CONST_ENQUEUE* outputs, void* workspace, cudaStream_t stream) TRT_NOEXCEPT override;

        virtual size_t getSerializationSize() const TRT_NOEXCEPT override;

        virtual void serialize(void* buffer) const TRT_NOEXCEPT override;

        bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const TRT_NOEXCEPT override {
            return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
        }

        const char* getPluginType() const TRT_NOEXCEPT override;

        const char* getPluginVersion() const TRT_NOEXCEPT override;

        void destroy() TRT_NOEXCEPT override;

        IPluginV2IOExt* clone() const TRT_NOEXCEPT override;

        void setPluginNamespace(const char* pluginNamespace) TRT_NOEXCEPT override;

        const char* getPluginNamespace() const TRT_NOEXCEPT override;

        DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const TRT_NOEXCEPT override;

        bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const TRT_NOEXCEPT override;

        bool canBroadcastInputAcrossBatch(int inputIndex) const TRT_NOEXCEPT override;

        void attachToContext(
            cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) TRT_NOEXCEPT override;

        void configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) TRT_NOEXCEPT override;

        void detachFromContext() TRT_NOEXCEPT override;

    private:
        void forwardGpu(const float* const* inputs, float *output, cudaStream_t stream, int batchSize = 1);
        int mThreadCount = 256;
        const char* mPluginNamespace;
        int mKernelCount;
        int mClassCount;
        int mYoloV5NetWidth;
        int mYoloV5NetHeight;
        int mMaxOutObject;
        std::vector<Yolo::YoloKernel> mYoloKernel;
        void** mAnchor;
    };

    class YoloPluginCreator : public IPluginCreator
    {
    public:
        YoloPluginCreator();

        ~YoloPluginCreator() override = default;

        const char* getPluginName() const TRT_NOEXCEPT override;

        const char* getPluginVersion() const TRT_NOEXCEPT override;

        const PluginFieldCollection* getFieldNames() TRT_NOEXCEPT override;

        IPluginV2IOExt* createPlugin(const char* name, const PluginFieldCollection* fc) TRT_NOEXCEPT override;

        IPluginV2IOExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) TRT_NOEXCEPT override;

        void setPluginNamespace(const char* libNamespace) TRT_NOEXCEPT override
        {
            mNamespace = libNamespace;
        }

        const char* getPluginNamespace() const TRT_NOEXCEPT override
        {
            return mNamespace.c_str();
        }

    private:
        std::string mNamespace;
        static PluginFieldCollection mFC;
        static std::vector<PluginField> mPluginAttributes;
    };
    REGISTER_TENSORRT_PLUGIN(YoloPluginCreator);
};

#endif 
