#test file

def get_iterator():
	return (i for i in iterates()), 1, 4

def iterates():
	x = range(100)
	for i in x:
		yield(i*2)


a, b, c = get_iterator()
print b, c
print a
for b in a:
	print b

