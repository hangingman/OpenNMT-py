# this should take list of indeces (delimited by blank lines) and file, retrieves lines with given
# indeces

import sys
print(sys.argv)
indeces = open(sys.argv[2]).readlines()
sentences = open(sys.argv[1]).readlines()
for i in indeces:
#	if not i:
	i=i.strip()
#	print(i)
	try: 
		#int(i)
		print(sentences[int(i)].strip())

	except ValueError:
		print()
	#else:
	#	print(sentences[int(i)])
