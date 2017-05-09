-- Test on images

-- Settings
-- Types:  light, vehicle, pedestrian and sign
pretype = 'sign'
-- Methods: DetectNet and SSD
local method = 'DetectNet'
-- Img size
im_h, im_w = 1024, 1024
-- Image folder
local folder_img = '/media/lym/leoymli/datasets/detection_datasets/TAD16K_2048/test/test_1/'
--------------------------------------------------------------------------------------


if (method == 'DetectNet') then
	dofile('detectnet.lua')
	print('Testing on DetectNet')
elseif (method == 'SSD') then
	dofile('ssd.lua')
	print('Testing on SSD')
else
	print('Unknow method!')
end

local filename = {}
for file in paths.iterfiles(folder_img) do
	table.insert(filename, file)
end
table.sort(filename)

for _, file in pairs(filename) do
	--file = '000136.jpg' -- test one specific image
	local start = sys.clock()
	local frame = image.load(folder_img..file)
	frame = image.scale(frame, im_w, im_h, 'simple')
	frame = process_one(frame)
	local time = sys.clock() - start
	--print("FPS: ".. 1/time)
	--print("Time: "..(time*1000)..'ms\n')

	-- Show results
	win = image.display{win=win,image=frame}

	-- Waiting for input to continue
	---[[
	  local re = io.read()
	  if re == 'n' then break end
	--]]
end

print('***Do predition finish***')