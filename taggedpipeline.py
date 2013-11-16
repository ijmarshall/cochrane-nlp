
import pipeline
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize.punkt import *
from collections import defaultdict, deque
import cPickle as pickle
from itertools import izip
from indexnumbers import swap_num

sent_tokenizer = PunktSentenceTokenizer()

class newPunktWordTokenizer(TokenizerI):
    """
    taken from new version of NLTK 3.0 alpha
    to allow for span tokenization of words (current 
    full version does not allow this)
    """
    def __init__(self, lang_vars=PunktLanguageVars()):
        self._lang_vars = lang_vars

    def tokenize(self, text):
        return self._lang_vars.word_tokenize(text)

    def span_tokenize(self, text):
        """
        Given a text, returns a list of the (start, end) spans of words
        in the text.
        """
        return [(sl.start, sl.stop) for sl in self._slices_from_text(text)]

    def _slices_from_text(self, text):
        last_break = 0
        contains_no_words = True
        for match in self._lang_vars._word_tokenizer_re().finditer(text):
            contains_no_words = False
            context = match.group()
            yield slice(match.start(), match.end())
        if contains_no_words:
            yield slice(0, 0) # matches PunktSentenceTokenizer's functionality

word_tokenizer = newPunktWordTokenizer()


with open('data/brill_pos_tagger.pck', 'rb') as f:
    pos_tagger = pickle.load(f)


class TaggedTextPipeline(pipeline.Pipeline):


    def __init__(self, text):
        self.functions = self.set_functions(text)
        self.load_templates()
        self.text = text




    def set_functions(self, text):

        text = swap_num(text.strip())
        
        tag_pattern = '<(\/?[a-z0-9_]+)>'
        
        # tag_matches_a is indices in annotated text
        tag_matches = [(m.start(), m.end(), m.group(1)) for m in re.finditer(tag_pattern, text)]

        tag_positions = defaultdict(list)
        displacement = 0 # initial re.finditer gets indices in the tagged text
                         # this corrects and produces indices for untagged text

        for start, end, tag in tag_matches:
            tag_positions[start-displacement].append(tag)
            displacement += (end-start) # add on the current tag length to cumulative displacement

        untagged_text = re.sub(tag_pattern, "", text) # now remove all tags

        sentences = []
        index_tag_stack = set() # tags active at current index
        

        char_stack = []
        current_word_tag_stack = []

        word_stack = []
        word_tag_stack = []

        sent_word_stack = []
        sent_tag_stack = []

        current_word_tags = []
        # per word tagging, so if beginning of word token is tagged only, e.g. '<n>Fifty</n>-nine'
        # and 'Fifty-nine' was a single token, then we assume the whole 

        keep_char = False # either keep or discard character; only kept if in a word token
        sent_indices, word_indices = self.wordsent_span_tokenize(untagged_text)

        i = 0

        while i < len(untagged_text):

            # first process tag stack to see whether next words are tagged
            for tag in tag_positions[i]:
                if tag[0] == '/':
                    try:
                        index_tag_stack.remove(tag[1:])
                    except:
                        print text
                        print untagged_text[i-20:i+20]
                        raise ValueError('unexpected tag %s in position %d of text' % (tag, i))
                else:
                    index_tag_stack.add(tag)


            if i == word_indices[0][1]: # if a word has ended
                keep_char = False
                word_stack.append(''.join(char_stack)) # push word to the word stack
                word_tag_stack.append(current_word_tag_stack)
                char_stack = [] # clear char stack
                current_word_tag_stack = []
                word_indices.popleft() # remove current word

            if i == word_indices[0][0]:
                current_word_tag_stack = list(index_tag_stack)
                keep_char = True

            if keep_char:
                char_stack.append(untagged_text[i])

            if i == sent_indices[0][1]:
                sent_word_stack.append(word_stack)
                sent_tag_stack.append(word_tag_stack)
                word_stack = []
                word_tag_stack = []

                sent_indices.popleft()

            i += 1

        base_functions = []
        
        # then pull altogether in a list of list of dicts
        # a list of sentences, each containing a list of word tokens,
        # each word represented by a dict
        for words, tags in izip(sent_word_stack, sent_tag_stack):

            base_sent_functions = []
            pos_tags = pos_tagger.tag(words)

            for (word, pos_tag), tag_list in izip(pos_tags, tags):
                base_word_functions = {"w": word,
                                       "p": pos_tag}
                for tag in tag_list:
                    base_word_functions["xml-annotation-[%s]" % (tag, )] = True

                base_sent_functions.append(base_word_functions)
            base_functions.append(base_sent_functions)

        return base_functions







    def wordsent_span_tokenize(self, text):
        """
        first sentence tokenizes then word tokenizes *per sentence*
        adjusts word indices for the full text
        this guarantees no overlap of words over sentence boundaries
        """

        sent_indices = deque(sent_tokenizer.span_tokenize(text)) # use deques since lots of left popping later
        word_indices = deque() # use deques since lots of left popping later

        for s_start, s_end in sent_indices:
            word_indices.extend([(w_start + s_start, w_end + s_start) for w_start, w_end in word_tokenizer.span_tokenize(text[s_start:s_end])])

        return sent_indices, word_indices
        
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
            words = {"BOW" + word["w"]: True for word in self.functions[i]}

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
                self.functions[i][j].update(words)

                # if pos is a noun, back fill the previous words
                pos = self.functions[i][j]["p"]
                if re.match("NN*", pos):
                    for k in range(last_noun_index, j):
                        self.functions[i][k]["next_noun"] = word
                    last_noun_index = j

            for k in range(last_noun_index, len(sent_function)):
                self.functions[i][k]["next_noun"] = "END_OF_SENTENCE"


def main():
    print "go"
    test = """
    Early inflammatory lesions and bronchial hyperresponsiveness are characteristics of the respiratory distress in premature neonates and are susceptible to aggravation by assisted ventilation. We hypothesized that treatment with <tx4_a>inhaled salbutamol and <tx3_a>beclomethasone</tx3_a></tx4_a> might be of clinical value in the prevention of bronchopulmonary dysplasia (BPD) in ventilator-dependent premature neonates. The study was double-blinded and <tx1_a><tx2_a>placebo</tx1_a></tx2_a> controlled. We studied <n>173</n> infants of less than 31 weeks of gestational age, who needed ventilatory support at the 10th postnatal day. They were randomised to four groups and received either <tx1>placebo + placebo</tx1>, <tx2>placebo + salbutamol</tx2>, <tx3>placebo + beclomethasone</tx3> or <tx4>beclomethasone + salbutomol</tx4>, respectively for 28 days. The major criteria for efficacy were: diagnosis of BPD (with score of severity), mortality, duration of ventilatory support and oxygen therapy. The trial groups were similar with respect to age at entry (9.8-10.1 days), gestational age (27.6-27.8 weeks), birth weight and oxygen dependence. We did not observe any significant effect of treatment on survival, diagnosis and severity of BPD, duration of ventilatory support or oxygen therapy. For instance, the odds-ratio (95% confidence interval) for severe or moderate BPD were 1.04 (0.52-2.06) for <tx3_a><tx4_a>inhaled beclomethasone</tx3_a></tx4_a> and 1.54 (0.78-3.05) for <tx4_a>inhaled salbutamol</tx4_a>. This randomised prospective trial does not support the use of treatment with inhaled <tx3_a><tx4_a>beclomethasone</tx3_a>, salbutamol</tx4_a> or their combination in the prevention of BPD in premature ventilated neonates."

    
    """

    # test = "Test of text <n>tagger</n>"

    b = TaggedTextPipeline(test)




if __name__ == '__main__':
    main()