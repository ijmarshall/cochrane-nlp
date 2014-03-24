#
#	custom CountVectorizer
#

from sklearn.feature_extraction.text import CountVectorizer


class ModularCountVectorizer(CountVectorizer):
	"""
	subclass of sklearn's CountVectorizer which allows for building up
	features with interaction terms
	"""