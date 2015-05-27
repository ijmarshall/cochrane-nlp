# simple string matching example using esmre python module
# which implements Aho-Corasick string matching
# https://code.google.com/p/esmre/

import esm

conditions = ['angina', 'migraine', 'depression', 'diabetes', 'unstable angina']

def build_index():

    index = esm.Index()

    for condition in conditions:
        index.enter(condition)

    index.fix()
    return index

def main():

    test_sentence = "We randomised 342 people with unstable angina to aspirin or placebo"

    index = build_index()

    print index.query(test_sentence)





if __name__ == '__main__':
    main()
