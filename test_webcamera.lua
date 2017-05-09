-- Test using web camera

-- Settings
require 'camera'
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

camera = image.Camera{idx=0, width=im_w, height=im_h, fps=30}

while (true) do
	local start = sys.clock()
	frame = camera:forward()
	frame = image.scale(frame, im_w, im_h, 'simple')
	frame = process_one(frame)
    win = image.display{win=win, image=frame}
    local time = sys.clock() - start
    print("FPS: ".. 1/time)
	print("Time: "..(time*1000)..'ms\n')
end

camera:stop()