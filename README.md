TAD16K: An enhanced dataset for autonomous driving (Source Code)
===================

Here is the source code of our paper ***TAD16K: An enhanced dataset for autonomous driving***.

We provide the evaluation results of two state-of-the-art object detection algorithms (SSD and DetectNet) on our dataset TAD16K, which can be used as the baseline for future comparison purpose. 

***SSD*** can be obtained at https://github.com/weiliu89/caffe/tree/ssd.

***DetectNet*** can be obtained at https://github.com/NVIDIA/DIGITS/tree/master/examples/object-detection.

We bind Caffe as a module in Torch7 and maintain an updated torch-caffe-binding source code at:
https://github.com/lymhust/caffe_torch_binding.

You have to have installed and built Caffe, then do this:

```bash
CAFFE_DIR=/*path-to-caffe-root*/ luarocks make
```
Then SSD and DetectNet can be easily evaluated under Torch7 environment.

**How to evaluate:**

1. Download `TAD16K` and our `pretrained models` at https://github.com/lymhust/caffe_torch_binding. 
2. Run `test_totxt.lua` to generate evaluation results of DetectNet and SSD.
3. Run `drawPRCurve_all.lua` to generate PR curve for each method.
4. Run `drawPRCurve_twoMethods.lua` to generate PR curve of two methods in one coordinate.

**Useful functions:**

* `test_basler.lua`:      test two methods using Basler industrial camera.
* `test_imgs.lua`:        test two methods using images stored in one folder.
* `test_webcamera.lua`:   test two methods using web camera.
* `json2txt.lua`:         parse the json file provided by TAD16K and save it as txt format.
* `detectnet.lua`:        parse the caffe model of DetectNet.
* `ssd.lua`:              parse the caffe model of SSD.
* `nms.lua`:              non maximum suppression.

**Examples:**
```lua
require 'caffe'

net = caffe.Net('deploy.prototxt', 'bvlc_alexnet.caffemodel', 'test')
input = torch.FloatTensor(10,3,227,227)
output = net:forward(input)

gradOutput = torch.FloatTensor(10,1000,1,1)
gradInput = net:backward(input, gradOutput)
```

User can also use it inside a network as nn.Module, for example:

```lua
require 'caffe'

model = nn.Sequential()
model:add(caffe.Net('deploy.prototxt', 'bvlc_alexnet.caffemodel', 'test'))
model:add(nn.Linear(1000,1))
```
