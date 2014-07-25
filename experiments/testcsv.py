#
#	test out PDF escaping
#

import csv

headers = [1, 2, 3, 4, 5]

text1 = """
This text contain's all sort's of weird punctuation, including commas, "quotes",	tabs and semi-colons;
"""

text2 = "short text"

with open('test.csv', 'wb') as f:
	w = csv.DictWriter(f, fieldnames=headers, escapechar='\\')
	w.writeheader()
	w.writerow({1: text1, 2: text2, 3:text1, 4:text2, 5:text2})