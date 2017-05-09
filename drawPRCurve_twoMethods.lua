require 'gnuplot'
gnuplot.setterm('wxt')
torch.setdefaulttensortype('torch.FloatTensor')

-- This script is to generate the PR curve using all the testing results (test_1 to test_7) for two methods
-- Before running this script, make sure that you have already run the drawPRCurve_all.lua first and got the
-- PR curves for DetectNet and SSD

-- Settings

-- Please download results from TAD16K website and put it into the same folder with this script
local result_folder = './results/TT10K/'

-- light, pedestrian, vehicle, sign
local pretype = 'vehicle'
--------------------------------------------------------------------------------------------------------------

-- Draw PR curve two methods
local method = {'DetectNet','SSD'}
local precision_all, recall_all, confidence_all = {}, {}, {}

for _, v in pairs(method) do
	local rocFileName = result_folder..v..'/'..pretype..'/test_all_PRcurve_'..pretype..'.txt'
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

	table.insert(precision_all, torch.Tensor(precision))
	table.insert(recall_all, torch.Tensor(recall))
	table.insert(confidence_all, torch.Tensor(confidence))
end


--gnuplot.epsfigure('prcurve.eps')
gnuplot.title('PR curve '..pretype)
gnuplot.xlabel('recall')
gnuplot.ylabel('precision')
gnuplot.grid(true)
gnuplot.axis('image')
gnuplot.movelegend('left','bottom')
gnuplot.axis{0,1,0,1}
gnuplot.plot({method[1],recall_all[1],precision_all[1],'-'},
			 {method[2],recall_all[2],precision_all[2],'-'})
--gnuplot.plotflush()
--]]

