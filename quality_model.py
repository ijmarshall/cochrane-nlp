

CORE_DOMAINS = ["Random sequence generation", "Allocation concealment", "Blinding of participants and personnel",
                "Blinding of outcome assessment", "Incomplete outcome data", "Selective reporting"]
                # "OTHER" is generated in code, not in the mapping file
                # see data/domain_names.txt for various other criteria
                # all of these are available via QualityQuoteReader



import cPickle as pickle
from tokenizer import sent_tokenizer
from journalreaders import PdfReader
from unidecode import unidecode
import re

def save_models(models, filename):
    with open(filename, 'wb') as f:
        pickle.dump(models, f)

def load_models(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def doc_demo(models, testfile="testdata/demo.pdf", test_mode=False):

    import color

    print "Document demo: " + testfile
    print "=" * 40
    print

    raw_text = PdfReader(testfile).get_text()
    text = unidecode(raw_text)
    text = re.sub('\n', ' ', text)

    # text_sents = sent_tokenizer.tokenize(text)
    # tokenize into sentences
    sents = sent_tokenizer.tokenize(text)



    domain_limiter = 1 if test_mode else len(CORE_DOMAINS) # only parse first domain in test mode

    
    for test_domain, doc_model, doc_vec, sent_model, sent_vec in zip(CORE_DOMAINS[:domain_limiter], *models):
        
        ####
        ## PART ONE - get the predicted sentences with risk of bias information
        ####

        # vectorize the sentences
        X_sents = sent_vec.transform(sents)

        # get predicted 1 / -1 for the sentences
        pred_sents = sent_model.predict(X_sents)

        # get the sentences which are predicted 1
        positive_sents = [sent for sent, pred in zip(sents, pred_sents) if pred==1]

        # make a single string per doc
        summary_text = " ".join(positive_sents)


        ####
        ##  PART TWO - integrate summarized and full text, then predict the document class
        ####

        doc_vec.builder_clear()
        doc_vec.builder_add_docs([text])
        doc_vec.builder_add_docs([summary_text], prefix="high-prob-sent-")

        X_doc = doc_vec.builder_transform()

        prediction = doc_model.predict(X_doc)[0]
        print "-" * 30
        print test_domain


        prediction = {1: "Low", -1: "Unknown or high"}[prediction]

        print prediction


        if prediction == "Low":
            text_color = color.GREEN
        elif prediction == "Unknown or high":
            text_color = color.YELLOW

        color.printout(prediction, text_color)

        print "-" * 30




def generate_models(test_mode=False):

    from quality3 import HybridDocModel, SentenceModel
    import numpy as np
    from sklearn.grid_search import GridSearchCV
    from sklearn.linear_model import SGDClassifier

    
    doc_models = [] # models will be stored in a list
    doc_vecs = []
    sent_models = []
    sent_vecs = []

    d = HybridDocModel(test_mode=test_mode)
    d.generate_data(binarize=True)
    
    domain_limiter = 1 if test_mode else len(CORE_DOMAINS) # only parse first domain in test mode

    for test_domain in CORE_DOMAINS[:domain_limiter]:

        print test_domain

        
        domain_uids = d.domain_uids(test_domain)
        no_studies = len(domain_uids)
        

        tuned_parameters = {"alpha": np.logspace(-4, -1, 10), "class_weight": [{1: i, -1: 1} for i in np.logspace(-1, 1, 10)]}
        clf = GridSearchCV(SGDClassifier(loss="hinge", penalty="L2"), tuned_parameters, scoring='precision')

        
        ### generate simple sentence model for the domain, to include in the hybrid model
        s = SentenceModel(test_mode=test_mode)
        s.generate_data(uid_filter=domain_uids)
        s.vectorize()
        sents_X, sents_y = s.X_domain_all(domain=test_domain), s.y_domain_all(domain=test_domain)
        sent_tuned_parameters = [{"alpha": np.logspace(-4, -1, 5)}, {"class_weight": [{1: i, -1: 1} for i in np.logspace(0, 2, 10)]}]
        sent_clf = GridSearchCV(SGDClassifier(loss="hinge", penalty="L2"), tuned_parameters, scoring='recall')
        sent_clf.fit(sents_X, sents_y)

        sent_models.append(sent_clf.best_estimator_)
        sent_vecs.append(s.vectorizer)

        ### pass the sentence model to the hybrid binary model
        d.set_sent_model(sent_clf, s.vectorizer)
        d.vectorize(test_domain)
        
        X_all, y_all = d.X_y_uid_filtered(domain_uids, test_domain)
        clf.fit(X_all, y_all)

        doc_models.append(clf.best_estimator_)
        doc_vecs.append(d.vectorizer)

    return doc_models, doc_vecs, sent_models, sent_vecs

def main():
    # from sklearn.linear_model import SGDClassifier
    # from sklearn.feature_extraction import DictVectorizer

    test_mode=True

    models = generate_models(test_mode=test_mode)

    save_models(models, 'data/test_models.pck')

    
    doc_demo(models, test_mode=test_mode)

    


if __name__ == '__main__':
    main()