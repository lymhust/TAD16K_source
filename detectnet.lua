require 'caffe'
require 'image'
require './utils/nms'
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(8)

--indx_img = 1

-- Functions
-- Preprocess an image before passing it to a Caffe model.
-- We need to rescale from [0, 1] to [0, 255], convert from RGB to BGR,
-- and subtract the mean pixel.
function rgb2bgr(img)
	local perm = torch.LongTensor({3, 2, 1})
	img = img:index(1, perm):mul(255)
	return img
end


function process_one(img_ori)
	-- Process one image
	local img = rgb2bgr(img_ori):float()
	
	--sys.tic()
	model_dec:forward(img:reshape(1, 3, im_h_crop, im_w))
    local out_pixel = model_dec:getBlobData(ind_pixel):clone()[1][1]
	local out_box = model_dec:getBlobData(ind_box):clone()[1]
	out_pixel[out_pixel:le(thresh_coverage)] = 0 -- light and sign: 0.05
	local maxval = out_pixel:max()
	--print('exec time dec forward = '..(sys.toc()*1000)..'ms')
	--win1 = image.display{win=win1, image=out_pixel}
	
--sys.tic()
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
			local bottom = math.min(torch.round(out_box[4][r][c]+r*16-15), im_h_crop-4)
			local w = right-left+1
			local h = bottom-top+1
			local score = out_pixel[r][c]
		
			if (left<right and top<bottom and w>thresh and h>thresh) then
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
			local indx = nms(bbox, thresh_nms, 5) -- light and sign: 0.2
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

				--[[
				if (label ~= nil) then
					-- Classify and draw
					-- sys.tic()
					local img_cls = img[{{},{top,bottom},{left,right}}]
					img_cls = image.scale(img_cls, 46, 46, 'simple')
					local s,out_cls = model_cls:forward(img_cls:reshape(1,3,46,46)):squeeze():max(1)
					s = s[1]
					out_cls = out_cls[1]
					-- print('exec time cls forward = '..(sys.toc()*1000)..'ms')
					if (label[out_cls] ~= 'other') then
						image.drawRect(img_ori, left, top, right, bottom, {lineWidth=4, color={0,255,0}, inplace=true})
						image.drawText(img_ori, string.format("%.2f\nw:%d\nh:%d\n%s", score, right-left+1, bottom-top+1, label[out_cls]), left, 1, {color={0,255,0}, bg={0,0,0}, size=2, inplace=true})	
					else
						--image.drawRect(img_ori, left, top, right, bottom, {lineWidth=4, color={0,0,0}, inplace=true})
						--image.drawText(img_ori, string.format("%.2f\nw:%d\nh:%d\n%s", score, right-left+1, bottom-top+1, label[out_cls]), left, 1, {color={255,0,0}, bg={0,0,0}, size=2, inplace=true})	
					end
				else
					--image.save('./pos_neg/'..indx_img..'.jpg', img_ori[{{},{top,bottom},{left,right}}])
					--indx_img = indx_img + 1
					-- Draw only
					--image.drawRect(img_ori, left, top, right, bottom, {lineWidth=4, color={0,255,0}, inplace=true})
					--image.drawText(img_ori, string.format("%.2f\nw:%d\nh:%d\n", score, right-left+1, bottom-top+1), left, 1, {color={0,255,0}, bg={0,0,0}, size=2, inplace=true})
				end
				--]]
			end		
		end
	end
--print(sys.toc()*1000)

	return img_ori, result
end
---------------------------------------------------------------------------

-- Label type
-- light, vehicle, pedestrian and sign
-- pretype = 'sign'
local caffe_dec_deploy_file = nil
local caffe_dec_model_file = nil
--local caffe_cls_deploy_file = nil
--local caffe_cls_model_file = nil
--label = nil

if (pretype == 'light') then
	caffe_dec_deploy_file = './models_DetectNet/model_light/light_tengfei/deploy.prototxt'
	caffe_dec_model_file = './models_DetectNet/model_light/light_tengfei/light_epoch42.caffemodel'
	--caffe_cls_deploy_file = './models_DetectNet/model_light/model_light_cls4_50K/deploy.prototxt'
	--caffe_cls_model_file = './models_DetectNet/model_light/model_light_cls4_50K/snapshot_iter_37640.caffemodel'
    --label = {'G','other','R','Y'}
    thresh = 20

elseif (pretype == 'vehicle') then
	caffe_dec_deploy_file = './models_DetectNet/model_car/carnet/deploy.prototxt'
	caffe_dec_model_file = './models_DetectNet/model_car/carnet/snapshot_iter_239475.caffemodel'
    --caffe_cls_deploy_file = './models_DetectNet/model_car/vehicle_cls/deploy.prototxt'
	--caffe_cls_model_file = './models_DetectNet/model_car/vehicle_cls/snapshot_iter_860.caffemodel'
    --label = {'car','other'}
    thresh = 50

elseif (pretype == 'pedestrian') then
	caffe_dec_deploy_file = './models_DetectNet/model_ped/pednet/deploy.prototxt'
	caffe_dec_model_file = './models_DetectNet/model_ped/pednet/snapshot_iter_116676.caffemodel'
	thresh = 30

elseif (pretype == 'sign') then
	caffe_dec_deploy_file = './models_DetectNet/model_sign/deploy.prototxt'
	caffe_dec_model_file = './models_DetectNet/model_sign/sign_v3.1-epoch8.caffemodel'
	--caffe_cls_deploy_file = './models_DetectNet/model_sign/model_sign_cls108_25K/deploy.prototxt'
	--caffe_cls_model_file = './models_DetectNet/model_sign/model_sign_cls108_25K/snapshot_iter_32880.caffemodel'
	--label = torch.load('./models_DetectNet/model_sign/model_sign_cls108_25K/labels.t7')
	thresh = 20

else
	print('Unknow type!!!')
end


-- Load detection and classification caffe net
caffe.Net.initGPUMemoryScope()
model_dec = caffe.Net(caffe_dec_deploy_file, caffe_dec_model_file, 'test')
if (label ~= nil) then
	model_cls = caffe.Net(caffe_cls_deploy_file, caffe_cls_model_file, 'test')
end
model_dec:setModeGPU()
ind_pixel = model_dec:getBlobIndx('coverage')
ind_box = model_dec:getBlobIndx('bboxes')

-- Image info
im_h, im_w = 512, 1024
topend = 1
bottomend = im_h
im_h_crop = bottomend-topend+1
model_dec:reshape(1, 3, im_h_crop, im_w)

-- Thresh
thresh_coverage = 0.05
thresh_nms = 0.5