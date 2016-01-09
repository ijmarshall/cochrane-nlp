import matplotlib.pyplot as plt

import wordcloud

def word_cloud(words, axes, title):
    wc = wordcloud.WordCloud().generate(words)
    axes.imshow(wc)
    axes.axis('off')
    plt.title(title)

def sprint(message):
    """Helper function for printing eye catching messages"""

    print '*'*5, message, '*'*5

def duplicate_words(pairs):
    """Helper function which yields a number of duplicated words proportional to
    their corresponding coefficients"""

    for coef, word in pairs:
        for _ in range(int(coef*100)):
            yield word
