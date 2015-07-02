#
#	pubmed library
#   ijm 9/12
#

import urllib
import socket
import xml.etree.cElementTree as ET
from cochranenlp.output import progressbar


socket.setdefaulttimeout(6)

class Pubmed:

    def __init__(self, raise_errors=True):
        self.raise_errors = raise_errors

    def _ET2unicode(self, ET_instance):
        "returns unicode of elementtree contents"
        return ET.tostring(ET_instance, method="text", encoding="utf-8").strip()

    def _ETfind(self, element_name, ET_instance):
        "finds (first) subelement, returns unicode of contents if present, else returns None"
        subelement = ET_instance.find(element_name)
        if subelement is not None:
            return self._ET2unicode(subelement)
        else:
            return None

    def efetch(self, ids, db = "pubmed", convert2text=True, summary=False, xml_key="PubmedArticle/MedlineCitation", email="mail@ijmarshall.com"):
        searchstring_l = {'db': db,
                          'id': ','.join(ids),
                          'retmode': "xml",
                          'email': email,
                          'tool': "python-pmlib"}
        searchstring = urllib.urlencode(searchstring_l) # encode html special characters in url

        for i in range(5):
            try:
                response = urllib.urlopen("http://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi" , searchstring) # open url with queries by POST
                tree = ET.fromstring(response.read())
                break
            except:
                print "Retrying %d" % (i, )

        if convert2text == True:
            return self._efetch_parse(tree)
        else:
            return tree.findall(xml_key)

    def esearch(self, term, db = "pubmed", retmax=None, retstart=0, email="mail@ijmarshall.com"):
        searchstring_l = {'db': db,
                          'term': term,
                          'retstart': retstart,
                          'email': email,
                          'tool': "python-pmlib"}
        if retmax is not None:
            searchstring_l['retmax'] = retmax
        searchstring = urllib.urlencode(searchstring_l) # encode html special characters in url

        # print searchstring

        for i in range(5):
            try:
                response = urllib.urlopen("http://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi" , searchstring) # open url with queries by POST
                text = response.read()
                tree = ET.fromstring(text)
                # print text
                break
            except:
                print "Retrying %d" % (i, )
        else:
            raise IOError("No response received from Pubmed")

        return self._esearch_parse(tree)


    def esummary(self, ids, db = "pubmed", retmax=None, retstart=0, email="mail@ijmarshall.com"):
        searchstring_l = {'db': db,
                          'id': ','.join(ids),
                          'retstart': retstart,
                          'version': '2.0',
                          'email': email,
                          'tool': "python-pmlib"}
        if retmax is not None:
            searchstring_l['retmax'] = retmax
        searchstring = urllib.urlencode(searchstring_l) # encode html special characters in url
        response = urllib.urlopen("http://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi" , searchstring) # open url with queries by POST
        tree = ET.fromstring(response.read())
        return self._esummary_parse(tree)


    def _esearch_parse(self, tree):
        "checks for errors and parses return values from pubmed.esearch"
        error = tree.find("ERROR")
        if error is not None and self.raise_errors:
            with open('err.log', 'wb') as f:
                f.write(self._ET2unicode(tree))
            raise(StandardError(self._ET2unicode(error)))
        output_dict = {"Count": self._ETfind("Count", tree),
                       "RetMax": self._ETfind("RetMax", tree),
                       "RetStart": self._ETfind("RetStart", tree),
                       "IdList": [self._ET2unicode(id_tree) for id_tree in tree.findall("IdList/Id")]}
        return output_dict

    def _efetch_parse(self, tree, retformat='text'):
        "checks for errors and parses return values from pubmed.efetch"
        error = tree.find("ERROR")
        if error is not None and self.raise_errors:

            raise(StandardError(self._ET2unicode(error)))
        articles = tree.findall("PubmedArticle/MedlineCitation")
        output_dicts = []

        for article in articles:
            output_dicts.append ({"PMID": self._ETfind("PMID", article),
                                  "Journal": self._ETfind("Article/Journal/Title", article),
                                  "ArticleTitle": self._ETfind("Article/ArticleTitle", article),
                                  "Abstract": self._ETfind("Article/Abstract", article)})

        return output_dicts

    def _esummary_parse(self, tree, retformat='text'):
        "checks for errors and parses return values from pubmed.efetch"
        error = tree.find("ERROR")
        if error is not None and self.raise_errors:
            raise(StandardError(self._ET2unicode(error)))
        articles = tree.findall("DocumentSummarySet/DocumentSummary")
        output_dicts = []

        for article in articles:
            output_dicts.append ({"PubType": [self._ET2unicode(flag) for flag in article.findall("PubType/flag")]})


        return output_dicts




class IterSearch():
    " class to iterate through esearch results without having to manage the url queries "
    def __init__(self, term, db = "pubmed", retmax=100):

        self.term = term
        self.db = db
        self.retmax = retmax
        self.buffer = {}
        self.pubmed = Pubmed()

    def itersearch(self, show_progress=False):

        count = int(self.pubmed.esearch(self.term, db=self.db, retmax=0)["Count"])

        if show_progress:
            p = progressbar.ProgressBar(count/self.retmax, timer=True)

        for i in range(0, count, self.retmax):
            if show_progress:
                p.tap()
            result = self.pubmed.esearch(self.term, db=self.db, retmax=self.retmax, retstart=i)
            for study_id in result["IdList"]:
                yield(study_id)


class IterFetch():
    " class to iterate through efetch results without having to manage the url queries "
    def __init__(self, ids, db = "pubmed", retmax=50, xml_key="PubmedArticle/MedlineCitation"):

        self.ids = ids
        self.db = db
        self.retmax = retmax
        self.xml_key = xml_key
        self.pubmed = Pubmed()


    def iterfetch(self):


        count = len(self.ids)

        # first element in list: 0
        # last element in list: len(ids) - 1
        for i in range(0, count, self.retmax):

            if i + self.retmax > count:
                partial_ids = self.ids[i:]
            else:
                partial_ids = self.ids[i:i+self.retmax]



            result = self.pubmed.efetch(partial_ids, db=self.db, convert2text=False, xml_key=self.xml_key)

            for abstract_xml in result:
                yield(abstract_xml)

class IterSummary():
    " class to iterate through efetch results without having to manage the url queries "
    def __init__(self, ids, db = "pubmed", retmax=50):

        self.ids = ids
        self.db = db
        self.retmax = retmax
        self.pubmed = Pubmed()

    def itersummary(self):

        count = len(self.ids)

        # first element in list: 0
        # last element in list: len(ids) - 1

        for i in range(0, count, self.retmax):

            if i + self.retmax > count:
                partial_ids = self.ids[i:]
            else:
                partial_ids = self.ids[i:i+self.retmax]

            result = self.pubmed.esummary(partial_ids, db=self.db)
            for entry in result:
                yield(entry)












def main():

    results = IterSummary(["14975573", "18021533", "21353478", "6583195", "999485"], retmax=10)


    for i in results.itersummary():
        print i



if __name__ == "__main__":
	main()
