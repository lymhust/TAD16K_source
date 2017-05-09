-- Test using basler camera

-- Settings
-- Types:  light, vehicle, pedestrian and sign
pretype = 'vehicle'
-- Methods: DetectNet and SSD
local method = 'DetectNet'
-- Img size
im_h, im_w = 1024, 1024
----------------------------------------------------------


-- Test on images
if (method == 'DetectNet') then
	dofile('detectnet.lua')
	print('Testing on DetectNet')
elseif (method == 'SSD') then
	dofile('ssd.lua')
	print('Testing on SSD')
else
	print('Unknow method!')
end

require 'basler'

local bs = basler.bs()

while (bs:isGrabbing()) do
	local start = sys.clock()
	local frame = bs:retrieveResult()
	frame = frame:permute(3,1,2)
	frame = image.scale(frame, im_w, im_h, 'simple')
	frame = process_one(frame)
    win = image.display{win=win, image=frame:index(1,torch.LongTensor{3, 2, 1})}
    local time = sys.clock() - start
    print("FPS: ".. 1/time)
	print("Time: "..(time*1000)..'ms\n')
end