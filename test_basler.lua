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

require 'basler'

local bs = basler.bs()

while (bs:isGrabbing()) do
	local start = sys.clock()
	local frame = bs:retrieveResult()
	frame = frame:permute(3,1,2)
	--frame = image.scale(frame, im_w, im_h, 'simple')
	--frame, result = process_one(frame[{{},{topend,bottomend},{}}], file)
    win = image.display{win=win, image=frame:index(1,torch.LongTensor{3, 2, 1})}
    local time = sys.clock() - start
    --print("FPS: ".. 1/time)
	--print("Time: "..(time*1000)..'ms\n')
end







