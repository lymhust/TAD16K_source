require 'caffe'
require 'image'
require './utils/nms'
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(8)

-- Settings
-- Label type: light, vehicle, pedestrian and sign
local caffe_dec_deploy_file = nil
local caffe_dec_model_file = nil

-- Thresh
local thresh_size = nil
local thresh_coverage = 0.05
local thresh_nms = 0.5

-- Model files
if (pretype == 'light') then
	caffe_dec_deploy_file = './models_DetectNet/model_light/light_tengfei/deploy.prototxt'
	caffe_dec_model_file = './models_DetectNet/model_light/light_tengfei/light_epoch42.caffemodel'
    thresh_size = 20
elseif (pretype == 'vehicle') then
	caffe_dec_deploy_file = './models_DetectNet/model_car/carnet/deploy.prototxt'
	caffe_dec_model_file = './models_DetectNet/model_car/carnet/snapshot_iter_239475.caffemodel'
    thresh_size = 50
elseif (pretype == 'pedestrian') then
	caffe_dec_deploy_file = './models_DetectNet/model_ped/pednet/deploy.prototxt'
	caffe_dec_model_file = './models_DetectNet/model_ped/pednet/snapshot_iter_116676.caffemodel'
	thresh_size = 30
elseif (pretype == 'sign') then
	caffe_dec_deploy_file = './models_DetectNet/model_sign/deploy.prototxt'
	caffe_dec_model_file = './models_DetectNet/model_sign/sign_v3.1-epoch8.caffemodel'
	thresh_size = 20
else
	print('Unknow type!!!')
end

-- Load detection caffe net
caffe.Net.initGPUMemoryScope()
local model_dec = caffe.Net(caffe_dec_deploy_file, caffe_dec_model_file, 'test')
model_dec:setModeGPU()
local ind_pixel = model_dec:getBlobIndx('coverage')
local ind_box = model_dec:getBlobIndx('bboxes')
model_dec:reshape(1, 3, im_h, im_w)
-------------------------------------------------------------------------------------------

-- Functions
-- Preprocess an image before passing it to a Caffe model.
-- We need to rescale from [0, 1] to [0, 255], convert from RGB to BGR
function rgb2bgr(img)
	local perm = torch.LongTensor({3, 2, 1})
	img = img:index(1, perm):mul(255)
	return img
end

function process_one(img_ori)
	-- Process one image
	local img = rgb2bgr(img_ori):float()
	
	model_dec:forward(img:reshape(1, 3, im_h, im_w))
    local out_pixel = model_dec:getBlobData(ind_pixel):clone()[1][1]
	local out_box = model_dec:getBlobData(ind_box):clone()[1]
	out_pixel[out_pixel:le(thresh_coverage)] = 0
	local maxval = out_pixel:max()
	
	local result = {}
	if (maxval > 0) then -- Detected
		local box_loc = out_pixel:nonzero()

		local bbox = torch.Tensor(box_loc:size(1), 5)
	    local indx = 0

		for i = 1, box_loc:size(1) do
			local r, c = box_loc[i][1], box_loc[i][2]
			local left = math.max(torch.round(out_box[1][r][c]+c*16-15), 4)
			local top = math.max(torch.round(out_box[2][r][c]+r*16-15), 4)
			local right = math.min(torch.round(out_box[3][r][c]+c*16-15), im_w-4)
			local bottom = math.min(torch.round(out_box[4][r][c]+r*16-15), im_h-4)
			local w = right-left+1
			local h = bottom-top+1
			local score = out_pixel[r][c]
		
			if (left<right and top<bottom and w>thresh_size and h>thresh_size) then
				indx = indx + 1
				bbox[indx][1] = left
				bbox[indx][2] = top
				bbox[indx][3] = right 
				bbox[indx][4] = bottom
				bbox[indx][5] = score
			end
		end
		
		if (indx > 0) then
			bbox = bbox[{{1,indx},{}}]	

			-- Merge box NMS and draw
			local indx = nms(bbox, thresh_nms, 5)
			local id = 0

			for i = 1, indx:size(1) do
				local left = bbox[indx[i]][1]
				local top = bbox[indx[i]][2]
				local right = bbox[indx[i]][3]
				local bottom = bbox[indx[i]][4]
				local score = bbox[indx[i]][5]
				table.insert(result, {left,top,right,bottom,score})
				image.drawRect(img_ori, left, top, right, bottom, {lineWidth=4, color={0,255,0}, inplace=true})
				image.drawText(img_ori, string.format("%.2f\nw:%d\nh:%d\n", score, right-left+1, bottom-top+1), left, 1, {color={0,255,0}, bg={0,0,0}, size=2, inplace=true})
			end		
		end
	end

	return img_ori, result
end
---------------------------------------------------------------------------
