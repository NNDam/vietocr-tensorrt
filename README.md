# VietOCR-TensorRT
<p align="center">
<img src="https://github.com/pbcquoc/vietocr/raw/master/image/sample.png" width="1000" height="300">
</p>

# Performance
VGG-Transformer | Baseline | TensorRT-FP16 
--- | --- | ---  
Time | 94.8 ms | 43.4 ms 
GPU Memory | 2480 Mb | 1613 Mb
Speed Up | 1.00x | 2.18x

# Requirements
* torch 1.7.0 + torchvision 0.8.0 (torch 1.8.0 & 1.8.1 not supported)
* onnx-simplifier
* TensorRT 7.2
* pycuda
```
sudo apt-get install build-essential python-dev python-setuptools libboost-python-dev libboost-thread-dev -y
sudo apt-get install python-numpy python3-numpy -y
sudo apt-get install libboost-all-dev
git clone --recursive --branch v2020.1 https://github.com/inducer/pycuda.git
cd pycuda
python configure.py --cuda-root=/usr/local/cuda-10.2
pip install -e .
```

# Convert to TensorRT (Example: VGG-Transformer)
### Step 1. Convert CNN + Transformer to ONNX
* Convert CNN + Transformer Encoder
```
    python convert_trt_encoder.py
```
* Convert Transformer Decoder
```
    python convert_trt_decoder.py
```
this will export model to 2 ONNX sub-models called transformer_decoder.onnx (CNN included) & transformer_encoder.onnx
### Step 2. Run ONNX simplifier
* Simply CNN + Transformer Encoder
```
    python -m onnxsim transformer_encoder.onnx transformer_encoder.onnx --dynamic-input-shape --input-shape=5,3,32,160
```
* Simply Transformer Decoder
```
    python -m onnxsim transformer_decoder.onnx transformer_decoder.onnx --dynamic-input-shape --input-shape tgt_inp:20,1 memory:170,1,256
```
### Step 3. Convert ONNX (simplified) to TensorRT with dynamic shape
Assume that min image size = 32x128, max image size = 32x768, max batch size = 32, max sequence length = 128
* Convert CNN + Transformer Encoder 
```
    /usr/src/tensorrt/bin/trtexec --explicitBatch \
                                --onnx=transformer_encoder.onnx \
                                --saveEngine=transformer_encoder.trt \
                                --minShapes=input:1x3x32x128 \
                                --optShapes=input:32x3x32x512 \
                                --maxShapes=input:32x3x32x768 \
                                --verbose \
                                --fp16
```
* Convert Transformer Decoder 
```
    /usr/src/tensorrt/bin/trtexec --explicitBatch \
                                --onnx=transformer_decoder.onnx \
                                --saveEngine=transformer_decoder.trt \
                                --minShapes=tgt_inp:1x1,memory:64x1x256 \
                                --optShapes=tgt_inp:64x32,memory:256x32x256 \
                                --maxShapes=tgt_inp:128x32,memory:384x32x256 \
                                --verbose \
                                --fp16
```
### Step 4. Run demo (batch inference)
```
    python trt_ocr_demo.py
```

# To do list
- [x] Convert VGG to TensorRT with dynamic shape 
- [x] Convert Transformer to TensorRT with dynamic shape 
- [x] Dynamic batch inference with TensorRT 
- [ ] Convert Sequence-To-Sequence to TensorRT with dynamic shape 
- [ ] Refactor & easy for user 

# Reference
- Original Source: https://github.com/pbcquoc/vietocr
