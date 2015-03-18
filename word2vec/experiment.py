from __future__ import unicode_literals

import logging

from spacy.en import English
from spacy.parts_of_speech import DET, NUM, PUNCT, X, PRT, NO_TAG, EOL

from gensim import models
from gensim.models import Word2Vec

import cochranenlp
from cochranenlp.readers import biviewer

from ftfy import fix_text

##### from https://github.com/oreillymedia/t-SNE-tutorial

# We'll use matplotlib for graphics.
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib

# We import seaborn to make nice plots.
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

def scatter(x, labels):
    le = preprocessing.LabelEncoder()
    le.fit(labels)

    colors = np.array([le.transform(label) for label in labels])

    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels
    txts = []
    for i, label in enumerate(le.classes_):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, label, fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts

######

from sklearn import preprocessing
from sklearn.manifold import TSNE

nlp = English()

word2vec_model = "/Volumes/Helios/vec-space-models/wikipedia-pubmed-and-PMC-w2v.bin"

viewer = biviewer.PDFBiViewer()

model = Word2Vec.load_word2vec_format(word2vec_model, binary=True)

n = 5000
exclude = [DET, NUM, PUNCT, X, PRT, NO_TAG, EOL]
domains = ["CHAR_PARTICIPANTS", "CHAR_OUTCOMES", "CHAR_INTERVENTIONS"]

domain_labels = {
    "CHAR_PARTICIPANTS": "P",
    "CHAR_OUTCOMES": "O",
    "CHAR_INTERVENTIONS": "I"
}

def normalize(s):
    return fix_text(s.decode("utf-8", "ignore"))

def is_eligable(tok):
    return tok.pos not in exclude and tok.norm_.isalnum()

def tokenize(s):
    return [tok.norm_ for tok in nlp(s) if is_eligable(tok)]

def embedding(model, tokens):
    vecs = [model[token] for token in tokens if model.vocab.has_key(token)]
    return np.average(np.asmatrix(vecs), 0).A[0,]

def get_fragments(domain, n):
    return [s[0]["CHARACTERISTICS"][domain] for s in itertools.islice(viewer, n)]

def fragment_embedding(model, fragment):
    tokens = tokenize(normalize(fragment))
    return embedding(model, tokens) if len(tokens) else np.zeros(200) # 200 is the dimensionality of the dense vector

fragments = [(domain, fragment) for domain in domains for fragment in get_fragments(domain,n) if fragment]

le = preprocessing.LabelEncoder()
le.fit(domains)

y = np.hstack(domain_labels[domain] for domain, fragment in fragments)

# the asfarray comes from this bug https://github.com/scikit-learn/scikit-learn/issues/4124
vecs = np.asfarray(np.vstack([fragment_embedding(model, fragment) for domain, fragment in fragments]), dtype='float')

proj = TSNE().fit_transform(vecs)

scatter(proj, y)
plt.savefig('tsne.png', dpi=300)
