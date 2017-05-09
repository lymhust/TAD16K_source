require 'gnuplot'
gnuplot.setterm('wxt')
torch.setdefaulttensortype('torch.FloatTensor')

-- This script is to generate the PR curve using all the testing results (test_1 to test_7) for one method
-- Settings
local method = 'DetectNet' --DetectNet or SSD

-- Please download results from TAD16K website and put it into the same folder with this script
local result_folder = './results/TT10K/'..method..'/'

-- Types:  light, vehicle, pedestrian and sign
local pretype = 'sign'
---------------------------------------------------------------------------------------------------------------


local tmp = {'label','result'}
local test = {'test_1','test_2','test_3','test_4','test_5','test_6','test_7'}
local filename = {}

-- Imagelist all
local tmpfilename = result_folder..'/test_all_label_imagelist.txt'
table.insert(filename, tmpfilename)
local file_tmp = io.open(tmpfilename, 'w')
for _, v in pairs(test) do
	for str in io.lines(result_folder..v..'_label_imagelist.txt') do
		file_tmp:write(str..'\n')
	end
	print(v)
end
file_tmp:close()
print('Imagelist all created')

-- Result all and label all
for _, v1 in pairs(tmp) do
	local tmpfilename = result_folder..pretype..'/test_all_'..v1..'_'..pretype..'.txt'
	table.insert(filename, tmpfilename)
	local file_tmp = io.open(tmpfilename, 'w')
	for _, v2 in pairs(test) do
		for str in io.lines(result_folder..pretype..'/'..v2..'_'..v1..'_'..pretype..'.txt') do
			file_tmp:write(str..'\n')
		end
		print(v2)
	end
	file_tmp:close()
end
print('Result all and label all created')


print(filename)
local rocFileName = result_folder..pretype..'/test_all_PRcurve_'..pretype..'.txt'
-- Evaluate result txt
---[[
local eva = require './utils/evaluate'
local listFile = filename[1]
local annotFile = filename[2]
local detFile = filename[3]
local matchScoreThreshold = 0.5;
local mAP = eva.evaluate(' ',listFile,annotFile,detFile,rocFileName,matchScoreThreshold)
--]]


-- Draw PR curve
---[[
require 'gnuplot'
gnuplot.setterm('wxt')
torch.setdefaulttensortype('torch.FloatTensor')


local precision, recall, confidence = {}, {}, {}
for str in io.lines(rocFileName) do
	--print(str)
	local tmp = string.split(str, ' ')
	local pre = tonumber(tmp[1])
	local rec = tonumber(tmp[2])
	local con = tonumber(tmp[3])
	if (rec>0 and pre>0) then
		table.insert(precision, pre)
		table.insert(recall, rec)
		table.insert(confidence,con)
	end
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

