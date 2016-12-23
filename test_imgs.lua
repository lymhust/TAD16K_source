-- Types:  light, vehicle, pedestrian and sign
pretype = 'pedestrian'
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

local folder_img = '/home/lym/mydisk/data/ped_images/'
local filename = {}
for file in paths.iterfiles(folder_img) do
	table.insert(filename, file)
end
table.sort(filename)

for _, file in pairs(filename) do
	file = '2015-03-26-12-51-24_01170.jpg'
	local start = sys.clock()
	local frame = image.load(folder_img..file)
	frame = image.scale(frame, im_w, im_h, 'simple')
	frame, result = process_one(frame[{{},{topend,bottomend},{}}], file)
	local time = sys.clock() - start
	print("FPS: ".. 1/time)
	print("Time: "..(time*1000)..'ms\n')

	-- Show results
	win = image.display{win=win,image=frame}
	--print('Continue? (y/n)')
	--local re = io.read()
	--if re == 'n' then break end
end
print('***Do predition finish***')