

s1 = "Long-term androgen suppression plus radiotherapy (AS+RT) is standard treatment of high-risk prostate cancer."

s2 = "To compare the test-retest reliability, convergent validity, and overall feasibility/ usability of activity-based (AB) and time-based (TB) approaches for obtaining self-reported moderate-to-vigorous physical activity (MVPA) from adolescents."

s3 = "This study was conducted to determine if prophylactic cranial irradiation (PCI) improves survival in locally advanced non-small-cell lung cancer (LA-NSCLC)"

s4 = "Alternatives to cytotoxic agents are desirable for patients with HIV-associated Kaposi's sarcoma (KS)."

s5 = "The primary objective was assessment of antitumor activity using modified AIDS Clinical Trial Group (ACTG) criteria for HIV-KS."

s6 = "To determine the effectiveness of bortezomib plus irinotecan and bortezomib alone in patients with advanced gastroesophageal junction (GEJ)and gastric adenocarcinoma."

import re

def make_dictionary(s):

	s_l = s.lower()

	# first get indices of abbreviations
	index_groups = [(m.start(0), m.end(0)) for m in re.finditer('\([a-z\+\-]+\)', s_l)]

	output = []

	lookup = {}

	for start_i, end_i in index_groups:

		abbreviation = re.sub('[^a-z]', '', s[start_i+1: end_i-1].lower())
		abbreviation_i = len(abbreviation)-1

		end_j = start_i-1
		span = None

		for i in range(end_j, 0, -1):

			if abbreviation_i == 0:
				# the first character of the abbreviation has to be the start
				# of a word (the others not necessarily)
				abbreviation_char = " " + abbreviation[abbreviation_i]
				text_char = s[i-1:i+1]
			else:
				abbreviation_char = abbreviation[abbreviation_i]
				text_char = s[i]

			if abbreviation_char == text_char:
				abbreviation_i -= 1
				if abbreviation_i == -1:
					span = i, end_j
					break
		output.append(span)
		lookup[s[start_i+1:end_i-1]] = s[span[0]:span[1]]

	return lookup



def main():
	s = s1
	d =  make_dictionary(s)
	print d


if __name__ == '__main__':
	main()






