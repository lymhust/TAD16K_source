-- Save test results to txt
-- Settings
-- Testing folders: test_1 to test_7
local folders = {'test_1','test_2','test_3','test_4',
                 'test_5','test_6','test_7'}
-- local folders = {'test_1'} --test one specific folder

-- Types:  light, vehicle, pedestrian and sign
pretype = 'sign'

-- Methods: DetectNet and SSD
local method = 'DetectNet'

-- TAD16K test folder
local testbase = '/media/lym/Work/data/TAD16K/test/'
----------------------------------------------------------------------

-- Load model
if (method == 'DetectNet') then
	dofile('detectnet.lua')
elseif (method == 'SSD') then
	dofile('ssd.lua')
else
	print('Unknow method!')
end


for _, val in pairs(folders) do

	-- Generate result txt
    local test_folder = val
	---[[
	local folder_img = testbase..test_folder..'/'
	local file_res = io.open('./results/TT10K/'..method..'/'..pretype..'/'..test_folder..'_result_'..pretype..'.txt', 'w')

	local filename = {}
	for str in io.lines('./results/TT10K/'..method..'/'..test_folder..'_label_imagelist.txt') do
		table.insert(filename, str)
	end

	for _, file in pairs(filename) do
		--local start = sys.clock()
		local frame = image.load(folder_img..file)
		frame = image.scale(frame, im_w, im_h, 'simple')
		frame, result = process_one(frame[{{},{topend,bottomend},{}}])
		file_res:write(file..'\n')
		file_res:write(#result..'\n')
		for _, v in pairs(result) do
			local left,top,right,bottom,score = v[1],v[2],v[3],v[4],v[5]
			local w, h = right-left+1, bottom-top+1
			file_res:write(left..' '..top..' '..w..' '..h..' '..score..'\n')
		end
		print(file)
		--local time = sys.clock() - start
		--print("FPS: ".. 1/time)
		--print("Time: "..(time*1000)..'ms\n')

		-- Show results
		--win = image.display{win=win,image=frame}
		--print('Continue? (y/n)')
		--local re = io.read()
		--if re == 'n' then break end
	end
	file_res:close()
	print('***Do predition finish***')
	--]]


	-- Evaluate result txt
	---[[
	local eva = require './utils/evaluate'
	local listFile = './results/TT10K/'..method..'/'..test_folder..'_label_imagelist.txt';
	local annotFile = './results/TT10K/'..method..'/'..pretype..'/'..test_folder..'_label_'..pretype..'.txt';
	local detFile = './results/TT10K/'..method..'/'..pretype..'/'..test_folder..'_result_'..pretype..'.txt';
	local rocFileName = './results/TT10K/'..method..'/'..pretype..'/'..test_folder..'_PRcurve_'..pretype..'.txt';
	local matchScoreThreshold = 0.5;
	local mAP = eva.evaluate(" ",listFile,annotFile,detFile,rocFileName,matchScoreThreshold)
	--]]


	-- Draw PR curve
	---[[
	require 'gnuplot'
	gnuplot.setterm('wxt')
	torch.setdefaulttensortype('torch.FloatTensor')


	local precision, recall, confidence = {}, {}, {}
	for str in io.lines(rocFileName) do
		print(str)
		local tmp = string.split(str, ' ')
		table.insert(precision, tonumber(tmp[1]))
		table.insert(recall, tonumber(tmp[2]))
		table.insert(confidence, tonumber(tmp[3]))
	end

	precision = torch.Tensor(precision)
	recall = torch.Tensor(recall)
	confidence = torch.Tensor(confidence)

	--gnuplot.epsfigure('prcurve.eps')
	gnuplot.title('PR curve '..pretype..', mAP: '..string.format('%.2f',mAP))
	gnuplot.xlabel('recall')
	gnuplot.ylabel('precision')
	gnuplot.grid(true)
	gnuplot.axis('image')
	gnuplot.movelegend('left','bottom')
	gnuplot.axis{0,1,0,1}
	gnuplot.plot({pretype,recall,precision,'-'})
	--gnuplot.plotflush()
	--]]

end