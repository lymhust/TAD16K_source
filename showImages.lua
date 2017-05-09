require 'image'
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(8)


-- Image info
im_h, im_w = 1024,1024--576, 1024


local folder = './data_manipulate/test/'
local filename = {}

for file in paths.iterfiles(folder) do
	table.insert(filename, file)
end

table.sort(filename)

local first = true
for _, file in pairs(filename) do
	local start = sys.clock()
	frame = image.scale(image.load(folder..file), im_w, im_h, 'simple')
	local time = sys.clock() - start
	win = image.display{win=win,image=frame}
	sys.sleep(0.05)
    print("FPS: ".. 1/time)
	print("Time: "..(time*1000)..'ms\n')
	--if (first) then sys.sleep(30) first = false end
end

