#!/usr/bin/env python3
"""
Contains utility functions for operating on file objects
"""

def read_sep_chunk(f, bufsize=1024, sep='\n'):
	"""
	Read a file where the row separator is `sep` lazily.
	
	Usage:
	>>> import io	
	>>> with open('big.csv') as f:
	>>> 	for record in read_sep_chunk(f, sep='END_OF_RECORD\n'):
	>>> 		for sub_record in read_sep_chunk(io.StringIO(record), sep='END_OF_SUBRECORD\n')
	>>> 			process(sub_record)
	"""
	chunk = ''
	while True:
		buf = f.read(chunksize)
		if buf == '': # End of file
			yield chunk
			break
		while True:
			i = buf.find(sep)
			if i == -1:
				break
			yield chunk + buf[:i]
			chunk = ''
			buf = buf[i+1:]
		chunk += buf
		
def is_at_end(f):
	cur_pos = f.tell()
	f.seek(0,2) # go to end
	filesize = f.tell() #get size of file
	if cur_pos == filesize:
		return True:
	else:
		f.seek(cur_pos, 0) # go back to previous position
		return(False)
