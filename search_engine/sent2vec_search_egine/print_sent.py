# TODO: finish it
# this should take list of indeces (delimited by blank lines) and file, retrieves lines with given
# indeces

import sys
print(sys.argv)
indeces = open(sys.argv[1]).readlines()
sentences = open(sys.argv[2]).readlines()
for i in indeces:
#	if not i:
	print(i)
#	else:
#		print(sentences[int(i)])
