TAD16K: An enhanced benchmark for autonomous driving (Source Code)
===================

Here is the source code of our paper ***TAD16K: An enhanced benchmark for autonomous driving***.

We provide the evaluation results of two state-of-the-art object detection algorithms (SSD and DetectNet) on our benchmark TAD16K, which can be used as the baseline for future comparison purpose. 

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

1. Download `TAD16K`, `pretrained models` and `results` at http://autopilot.qq.com/ICIP2017/. 
2. Replace the empty folders (`models_DetectNet`, `models_SSD` and `results`).
3. Run `test_totxt.lua` to generate evaluation results of DetectNet and SSD.
4. Run `drawPRCurve_all.lua` to generate PR curve for each method.
5. Run `drawPRCurve_twoMethods.lua` to generate PR curve of two methods in one coordinate.
* Before running, please goto 'utils/evaluate' and build the source: 

```bash
mkdir build
cd ./build
cmake ..
make
```

**Useful functions:**

* `test_basler.lua`:      test two methods using Basler industrial camera.
* `test_imgs.lua`:        test two methods using images stored in one folder.
* `test_webcamera.lua`:   test two methods using web camera.
* `detectnet.lua`:        parse the caffe model of DetectNet.
* `ssd.lua`:              parse the caffe model of SSD.
* `nms.lua`:              non maximum suppression.
* `json2txt.lua`:         parse the json file provided by TAD16K and save it as txt format.
* `data_statistics.lua`:  statistics of the four types of objects in TAD16K.

**Examples:**
```lua
require 'camera'
-- Types:  light, vehicle, pedestrian and sign
pretype = 'sign'
-- Methods: DetectNet and SSD
local method = 'DetectNet'

if (method == 'DetectNet') then
	dofile('detectnet.lua')
elseif (method == 'SSD') then
	dofile('ssd.lua')
else
	print('Unknow method!')
end

-- Testing using web camera
camera = image.Camera{idx=0, width=im_w, height=im_h, fps=30}

while (true) do
  local start = sys.clock()
  frame = camera:forward()
  frame = image.scale(frame, im_w, im_h, 'simple')
  frame, result = process_one(frame[{{},{topend,bottomend},{}}])
  win = image.display{win=win, image=frame}
  local time = sys.clock() - start
  print("FPS: ".. 1/time)
  print("Time: "..(time*1000)..'ms\n')
end
```


**DetectNet results on PointGrey frames:**


![](/result_imgs/sign.png)
![](/result_imgs/car.png)
![](/result_imgs/light.png)
![](/result_imgs/ped.png)
