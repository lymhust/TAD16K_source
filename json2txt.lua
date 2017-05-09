json = require 'json'
torch.setnumthreads(8)

--[[
local files = {'test_1_label','test_2_label','test_3_label','test_4_label',
			   'test_5_label','test_6_label','test_7_label'}
--]]

--[[
local files = {'train_1_label','train_2_label','train_3_label','train_4_label',
			   'train_5_label','train_6_label','train_7_label','train_8_label',
			   'train_9_label','train_10_label','train_11_label','train_12_label','train_13_label'}
--]]

---[[
local files = {'other_1_label','other_2_label','other_3_label','other_4_label',
			   'other_5_label','other_6_label','other_7_label','other_8_label',
			   'other_9_label','other_10_label','other_11_label','other_12_label',
			   'other_13_label', 'other_14_label', 'other_15_label', 'other_16_label'}
--]]

local type_need = 'sign'

if (type_need == 'light') then
    thresh = 0--20
elseif (type_need == 'vehicle') then
    thresh = 0--50
elseif (type_need == 'pedestrian') then
	thresh = 0--30
elseif (type_need == 'sign') then
	thresh = 0--20
else
	print('Unknow type!!!')
end

for _,v in pairs(files) do

	local file_json = v
	local file_addr = '/media/lym/Work/data/TT10K_2048/标注结果/other/'..file_json..'.json'
	local result_addr = '/media/lym/Work/code/nvidia_demo/detectnet_deploy/results/TT10K/labels_all/'

	local txt_file = io.open(result_addr..type_need..'/'..file_json..'_'..type_need..'.txt', 'w')
	local label = json.load(file_addr)
	local filename = io.open(result_addr..file_json..'_imagelist'..'.txt', 'w')
	local im_h_ori, im_w_ori = 2048, 2048
	local im_h_new, im_w_new = 1024, 1024

	for num = 1, #label do
		local tmp = label[num]
		local name = string.split(tmp.filename, '/')
		name = name[#name]
		local output = {}
		local annotation = tmp.annotations

		-- Process one image label
		for id_anno = 1, #annotation do
			local tmp_anno = annotation[id_anno]
			if (string.find(tmp_anno.type, type_need)~=nil) then
				local left =  math.max(tmp_anno.x, 1)
				local top = math.max(tmp_anno.y, 1)
				local w = tmp_anno.width
				local h = tmp_anno.height
				local right = math.min(left+w-1, im_w_ori)
				local bottom = math.min(top+h-1, im_h_ori)

				-- Rescale and filter
				left = left/im_w_ori*im_w_new
				top = top/im_h_ori*im_h_new
				right = right/im_w_ori*im_w_new
				bottom = bottom/im_h_ori*im_h_new
				w = right-left+1
				h = bottom-top+1

				if (left<right and top<bottom and w>thresh and h>thresh) then -- light and sign: 20; vehicle and pedestrian:50
					table.insert(output, {left,top,right-left+1,bottom-top+1,1})
				end
			end
		end
		------------------------------
		
		-- Write to txt
		txt_file:write(name..'\n')
		txt_file:write(#output..'\n')
		for id = 1, #output do
			txt_file:write(output[id][1]..' '..output[id][2]..' '..
						   output[id][3]..' '..output[id][4]..' '..output[id][5]..'\n')
		end
		filename:write(name..'\n')
	end		

	txt_file:close()
	filename:close()
	print(v)

end
print('***Finish_'..type_need..'***')