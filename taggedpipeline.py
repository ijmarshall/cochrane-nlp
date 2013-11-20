
import pipeline

# from nltk.tokenize import TreebankWordTokenizer

from collections import defaultdict, deque
import cPickle as pickle
from itertools import izip
from indexnumbers import swap_num
import re


from tokenizer import tag_words


with open('data/brill_pos_tagger.pck', 'rb') as f:
    pos_tagger = pickle.load(f)



class TaggedTextPipeline(pipeline.Pipeline):


    def __init__(self, text):
        self.text = re.sub('[nN]=([1-9]+[0-9]*)', r'N = \1', text)
        self.text = swap_num(text)
        self.functions = self.set_functions(self.text)
        self.load_templates()
        


    def set_functions(self, tagged_text):
        
        tag_tuple_sents = tag_words(tagged_text)

        base_functions = []

        # then pull altogether in a list of list of dicts
        # a list of sentences, each containing a list of word tokens,
        # each word represented by a dict
        for sent in tag_tuple_sents:

            base_sent_functions = []

            pos_tags = pos_tagger.tag([word for word, tag_list in sent])

            for (word, pos_tag), (word, tag_list) in izip(pos_tags, sent):
                base_word_functions = {"w": word,
                                       "p": pos_tag,
                                       "tags": []}
                for tag in tag_list:
                    base_word_functions["tags"].append(tag)

                base_sent_functions.append(base_word_functions)
            base_functions.append(base_sent_functions)

        return base_functions



        # [[{"w": word, "p": pos} for word, pos in pos_tagger.tag(self.word_tokenize(sent))] for sent in self.sent_tokenize(swap_num(text))]


    def load_templates(self):
        self.templates = (
                          (("w", 0),),
                          (("w", 1),),
                          (("w", 2),),
                          (("w", 3),),
                          (("w", -1),),
                          (("w", -2),),
                          (("w", -3),),
                          (('w', -2), ('w',  -1)),
                          # (('stem', -1), ('stem',  0)),
                          # (('stem',  0), ('stem',  1)),
                          (('w',  1), ('w',  2)),
                          # (('p',  0), ('p',  1)),
                          (('p',  1), ('p',  2)),
                          (('p',  -1), ('p',  -2)),
                          # (('stem', -2), ('stem',  -1), ('stem',  0)),
                          # (('stem', -1), ('stem',  0), ('stem',  1)),
                          # (('stem', 0), ('stem',  1), ('stem',  2)),
                          (('p', -2), ),
                          (('p', -1), ),
                          (('p', 1), ),
                          (('p', 2), ),
                          (('num', -1), ),
                          (('num', 1), ),
                          (('cap', -1), ),
                          (('cap', 1), ),
                          (('sym', 0), ),
                          (('sym', -1), ),
                          (('sym', 1), ),
                          # (('p1', 1), ),
                          # (('p2', 1), ),
                          # (('p3', 1), ),
                          # (('p4', 1), ),
                          # (('s1', 1), ),
                          # (('s2', 1), ),
                          # (('s3', 0), ),
                          # (('s4', 0), ),
                          (('wi', 0), ),
                          (('si', 0), ),
                          (('next_noun', 0), ),
                          )

        self.answer_key = lambda x: x["w"]

    def run_functions(self, show_progress=False):
        for i, sent_function in enumerate(self.functions):

            # line below not used yet
            # need to implement in Pipeline run_templates
            # words = {"BOW" + word["w"]: True for word in self.functions[i]}

            last_noun_index = 0

            for j, function in enumerate(sent_function):
                word = self.functions[i][j]["w"]
                features = {"num": word.isdigit(), # all numeric
                            "cap": word[0].isupper(), # starts with upper case
                            "sym": not word.isalnum(), # contains a symbol anywhere
                            "p1": word[0],  # first 1 char (prefix)
                            "p2": word[:2], # first 2 chars
                            "p3": word[:3], # ...
                            "p4": word[:4],
                            "s1": word[-1],  # last 1 char (suffix)
                            "s2": word[-2:], # last 2 chars
                            "s3": word[-3:], # ...
                            "s4": word[-4:],
                            # "stem": self.stem.stem(word),
                            "wi": j,
                            "si": i,
                            "punct": not any(c.isalnum() for c in word) # all punctuation}
                           }
                self.functions[i][j].update(features)

                # line below not used yet
                # need to implement in Pipeline run_templates
                # self.functions[i][j].update(words)

                # if pos is a noun, back fill the previous words
                pos = self.functions[i][j]["p"]
                if re.match("NN*", pos):
                    for k in range(last_noun_index, j):
                        self.functions[i][k]["next_noun"] = word
                    last_noun_index = j

            for k in range(last_noun_index, len(sent_function)):
                self.functions[i][k]["next_noun"] = "END_OF_SENTENCE"

    @pipeline.filters
    def get_tags(self):
        return [[{k: w[k] for k in ('w', 'tags')} for w in s] for s in self.get_base_functions()]

def main():

    test_text = """
    Early inflammatory lesions and bronchial hyperresponsiveness are characteristics of the respiratory distress
    in premature neonates and are susceptible to aggravation by assisted ventilation. We hypothesized that
    treatment with <tx4_a>inhaled salbutamol and <tx3_a>beclomethasone</tx3_a></tx4_a> might be of clinical
    value in the prevention of bronchopulmonary dysplasia (BPD) in ventilator-dependent premature neonates. The
    study was double-blinded and <tx1_a><tx2_a>placebo</tx1_a></tx2_a> controlled. We studied 1<n>7</n>3 infants
    of less than 31 weeks of gestational age, who needed ventilatory support at the 10th postnatal day. They
    were randomised to four groups and received either <tx1>placebo + placebo</tx1>, <tx2>placebo + salbutamol</tx2>
    , <tx3>placebo + beclomethasone</tx3> or <tx4>beclomethasone + salbutomol</tx4>, respectively for 28 days.
     The major criteria for efficacy were: diagnosis of BPD (with score of severity), mortality, duration of
      ventilatory support and oxygen therapy. The trial groups were similar with respect to age at entry (9.8-10.1
     days), gestational age (27.6-27.8 weeks), birth weight and oxygen dependence. We did not observe any
    significant effect of treatment on survival, diagnosis and severity of BPD, duration of ventilatory support
     or oxygen therapy. For instance, the odds-ratio (95% confidence interval) for severe or moderate BPD were
     1.04 (0.52-2.06) for <tx3_a><tx4_a>inhaled beclomethasone</tx3_a></tx4_a> and 1.54 (0.78-3.05) for <tx4_a>
     inhaled salbutamol</tx4_a>. This randomised prospective trial does not support the use of treatment with
     inhaled <tx3_a><tx4_a>beclomethasone</tx3_a>, salbutamol</tx4_a> or their combination in the prevention of
     BPD in premature ventilated neonates.
    """


    p = TaggedTextPipeline(test_text)

    tags = p.get_tags(flatten=True) # returns a list of tags

    print "example of tags format (word3 20-25):"
    print tags[25:30]

    print
    print "intervention 4 (words tagged tx4)"
    print [w["w"] for w in tags if "tx4" in w["tags"]]

    print
    print "number of people randomised"
    print [w["w"] for w in tags if "n" in w["tags"]]

if __name__ == '__main__':
    main()
