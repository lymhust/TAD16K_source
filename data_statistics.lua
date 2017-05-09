require 'image'

local pretype = 'vehicle'
local folder_lab = '/media/lym/Work/code/nvidia_demo/detectnet_deploy/results/TT10K/labels_all/'..pretype..'/'


local totalnum = 0
local wall, hall = {}, {}
for file in paths.iterfiles(folder_lab) do
	print(file)
	local file_lab = io.open(folder_lab..file, 'r')

	while (true) do
		local name = file_lab:read()
		if (name == nil) then break end
		file_lab:seek('cur',0)
		local num = tonumber(file_lab:read())

		if (num > 0) then
			for i = 1, num do
				file_lab:seek('cur',0)
				local info = file_lab:read()
				info = string.split(info, ' ')
				local left = math.max(torch.round(tonumber(info[1])), 1)
				local top = math.max(torch.round(tonumber(info[2])), 1)
				local right = math.min(left+torch.round(tonumber(info[3]))-1, 1024)
				local bottom = math.min(top+torch.round(tonumber(info[4]))-1, 1024)
				local w = tonumber(info[3])
				local h = tonumber(info[4])
				local label = info[5]
				if (left<right and top<bottom) then
					table.insert(wall, w*2)
					table.insert(hall, h*2)
					totalnum = totalnum+1
				end	
			end
		end
	end

	file_lab:close()
end

wall = torch.Tensor(wall)
hall = torch.Tensor(hall)
print(pretype..' num: '..totalnum)
print('W min:'..wall:min())
print('W max:'..wall:max())
print('H min:'..hall:min())
print('H max:'..hall:max())




