local ffi = require 'ffi'

ffi.cdef[[
double evaluate(const char *bDir, const char *lFile, const char *aFile, const char *dFile, const char *rFile, double matchScoreThreshold);
]]

local C = ffi.load('./utils/evaluate/build/libevaluate.so')

eva = {}

function eva.evaluate(bDir, lFile, aFile, dFile, rFile, thresh)
	assert(type(bDir) == 'string')
	assert(type(lFile) == 'string')
	assert(type(aFile) == 'string')
	assert(type(dFile) == 'string')
	assert(type(rFile) == 'string')
	assert(type(thresh) == 'number')
	return C.evaluate(bDir, lFile, aFile, dFile, rFile, thresh)
end

return eva