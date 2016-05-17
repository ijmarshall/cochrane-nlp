from nltk import word_tokenize


def abstract2idxs_generator(abstract, word2idx):
    for word in word_tokenize(abstract):

        yield word2idx[word]

def length_generator(abstracts):
    for abstract in abstracts:
        words = word_tokenize(abstract)

        yield len(words)
