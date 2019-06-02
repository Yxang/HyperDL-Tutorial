### 在CPP下使用TVM来部署mxnet模型（以Insightface为例）

自从AI被炒作以来，各个深度学习框架层出不穷。我们通常来讲，作为AI从业者，我们通常经历着标注-训练-部署的过程。其中部署是较为痛苦的工作，尤其是在跨平台如（移动端需要native对接的时候。）当然用于inference框架同样也是层出不穷。但是大多数框架框架往往性能都一般，或者缺少相关op，或者就是转换模型较为困难。TVM的出现很大程度上为模型部署带来了福音。

但是网上将用于TVM部署的教程还比较少，尤其是通过cpp和移动端部署。本文以Insightface Model Zoo中的MobileFaceNet为例，介绍一下如何编译Mxnet模型、在python下inference、在cpp下inference、对比人脸余弦距离、以及在android下的部署。

### 安装

TVM编译环境的安装需要LLVM编译器，可以简要遵循官方的教程。 [official installation tutorial](https://docs.tvm.ai/install/from_source.html#build-the-shared-library).

LLVM 7.0 可能会导致编译错误，推荐使用LLVM 6.0.1

#### 编译模型

TVM使用了一系列的优化措施来优化计算图，当模型编译完之后会生成若干个编译好的文件。在编译前要指定预编译的平台、架构、指令集等参数。

```python
import numpy as np
import nnvm.compiler
import nnvm.testing
import tvm
from tvm.contrib import graph_runtime
import mxnet as mx
from mxnet import ndarray as nd

prefix,epoch = "emore1",0
sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
image_size = (112, 112)
opt_level = 3

shape_dict = {'data': (1, 3, *image_size)}
target = tvm.target.create("llvm -mcpu=haswell")
# "target" means your target platform you want to compile.

#target = tvm.target.create("llvm -mcpu=broadwell")
nnvm_sym, nnvm_params = nnvm.frontend.from_mxnet(sym, arg_params, aux_params)
with nnvm.compiler.build_config(opt_level=opt_level):
   graph, lib, params = nnvm.compiler.build(nnvm_sym, target, shape_dict, params=nnvm_params)
lib.export_library("./deploy_lib.so")
print('lib export succeefully')
with open("./deploy_graph.json", "w") as fo:
   fo.write(graph.json())
with open("./deploy_param.params", "wb") as fo:
   fo.write(nnvm.compiler.save_param_dict(params))
```

运行该代码后会生成三个文件分别为deploy_lib.so 、deploy_graph.json 、deploy_param.params 。其中deploy_lib.so 为编译好的动态库，deploy_graph.json为部署使用的计算图、deploy_param.params为模型参数。

#### 使用TVM Python Runtime 进行简单的测试

TVM的Runtime(运行时)并不需要任何依赖，直接clone tvm后 make runtime.即可。

```python
import numpy as np
import nnvm.compiler
import nnvm.testing
import tvm
from tvm.contrib import graph_runtime
import mxnet as mx
from mxnet import ndarray as nd

ctx = tvm.cpu()
# load the module back.
loaded_json = open("./deploy_graph.json").read()
loaded_lib = tvm.module.load("./deploy_lib.so")
loaded_params = bytearray(open("./deploy_param.params", "rb").read())

input_data = tvm.nd.array(np.random.uniform(size=data_shape).astype("float32"))

module = graph_runtime.create(loaded_json, loaded_lib, ctx)
module.load_params(loaded_params)

# Tiny benchmark test.
import time
for i in range(100):
   t0 = time.time()
   module.run(data=input_data)
   print(time.time() - t0)
```

#### 使用C++来推理MobileFaceNet人脸识别模型

在C++下 TVM Runtime（运行时）仅仅需要编译时输出的so文件，包含  “tvm_runtime_pack.cc” 。runtime的体积也比较小，只有几百K。

下列的CPP代码包含了通过输入一张对齐后的人脸识别照片，输出归一化的之后的人脸向量。

```cpp
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>

class FR_MFN_Deploy{
    
    private:
        void * handle;
    
    public:
        FR_MFN_Deploy(std::string modelFolder)
        {
    
            tvm::runtime::Module mod_syslib = tvm::runtime::Module::LoadFromFile(modelFolder + "/deploy_lib.so");
            //load graph
            std::ifstream json_in(modelFolder + "/deploy_graph.json");
            std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
            json_in.close();
    
            int device_type = kDLCPU;
            int device_id = 0;
            // get global function module for graph runtime
            tvm::runtime::Module mod = (*tvm::runtime::Registry::Get("tvm.graph_runtime.create"))(json_data, mod_syslib, device_type, device_id);
            this->handle = new tvm::runtime::Module(mod);
    
            //load param
            std::ifstream params_in(modelFolder + "/deploy_param.params", std::ios::binary);
            std::string params_data((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
            params_in.close();
    
            TVMByteArray params_arr;
            params_arr.data = params_data.c_str();
            params_arr.size = params_data.length();
            tvm::runtime::PackedFunc load_params = mod.GetFunction("load_params");
            load_params(params_arr);
        }
    
        cv::Mat forward(cv::Mat inputImageAligned)
        {
            //mobilefacnet preprocess has been written in graph.
            cv::Mat tensor = cv::dnn::blobFromImage(inputImageAligned,1.0,cv::Size(112,112),cv::Scalar(0,0,0),true);
            //convert uint8 to float32 and convert to RGB via opencv dnn function
            DLTensor* input;
            constexpr int dtype_code = kDLFloat;
            constexpr int dtype_bits = 32;
            constexpr int dtype_lanes = 1;
            constexpr int device_type = kDLCPU;
            constexpr int device_id = 0;
            constexpr int in_ndim = 4;
            const int64_t in_shape[in_ndim] = {1, 3, 112, 112};
            TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &input);//
            TVMArrayCopyFromBytes(input,tensor.data,112*3*112*4);
            tvm::runtime::Module* mod = (tvm::runtime::Module*)handle;
            tvm::runtime::PackedFunc set_input = mod->GetFunction("set_input");
            set_input("data", input);
            tvm::runtime::PackedFunc run = mod->GetFunction("run");
            run();
            tvm::runtime::PackedFunc get_output = mod->GetFunction("get_output");
            tvm::runtime::NDArray res = get_output(0);
            cv::Mat vector(128,1,CV_32F);
            memcpy(vector.data,res->data,128*4);
            cv::Mat _l2;
            // normlize 
            cv::multiply(vector,vector,_l2);
            float l2 =  cv::sqrt(cv::sum(_l2).val[0]);
            vector = vector / l2;
            TVMArrayFree(input);
            return vector;
    }

};
```

我们可以通过输入两张对齐后的人脸照片来提取人脸向量。

```cpp
cv::Mat A = cv::imread("/Users/yujinke/Desktop/align_id/aligned/20171231115821836_face.jpg");
cv::Mat B = cv::imread("/Users/yujinke/Desktop/align_id/aligned/20171231115821836_idcard.jpg");
FR_MFN_Deploy deploy("./models");
cv::Mat v2 = deploy.forward(B);
cv::Mat v1 = deploy.forward(A);
```

测量余弦相似度

```cpp
inline float CosineDistance(const cv::Mat &v1,const cv::Mat &v2){
    return static_cast<float>(v1.dot(v2));
}
std::cout<<CosineDistance(v1,v2)<<std::endl;
```

简单的配置一个cmake文件

```bash
cmake_minimum_required(VERSION 3.6)
project(tvm_mobilefacenet)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -ldl -lpthread")
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY  ${CMAKE_CURRENT_SOURCE_DIR})
SET(HOME_TVM /Users/jackyu/downloads/tvm-0.5)
find_package(OPENCV REQUIRED)

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${HOME_TVM}/include)
INCLUDE_DIRECTORIES(${HOME_TVM}/3rdparty/dmlc-core/include)
INCLUDE_DIRECTORIES(${HOME_TVM}/3rdparty/dlpack/include)

add_executable(tvm_mobilefacenet  tvm_runtime_pack.cc main.cpp)
target_link_libraries(tvm_mobilefacenet    ${OpenCV_LIBS})
```

#### Todo：如何在在Android下部署整套人脸识别流程



