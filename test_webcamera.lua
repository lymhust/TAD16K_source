require 'camera'
-- Types:  light, vehicle, pedestrian and sign
pretype = 'vehicle'
-- Methods: DetectNet and SSD
local method = 'DetectNet'


-- Test on images
if (method == 'DetectNet') then
	dofile('detectnet.lua')
elseif (method == 'SSD') then
	dofile('ssd.lua')
else
	print('Unknow method!')
end


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

camera:stop()














