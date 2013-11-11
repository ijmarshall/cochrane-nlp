#  Journal readers
#  Iain
#  2012-12-08

#
#  *** REQUIRES UNIX/LINUX OS WITH pdftotext INSTALLED ***
#

import os
import glob
import codecs
#from bs4 import BeautifulSoup
import BeautifulSoup
import subprocess

class TextReader:
    """Reads plain text files, and base class for other file types

    >>> t = TextReader('testdata/simpletest/textreader.txt')
    >>> print t.get_text()
    Test Journal Article
    by the authors

    """
    def __init__(self, filename):
        self.data = {"filetype": "text",
                     "raw": self.import_text(filename)}

    def import_text(self, filename):
        "gets unicode utf-8 file contents"
        with codecs.open(filename, "rb", "utf-8") as f:
            raw_text = f.read()
        return raw_text.strip()

    def get_text(self):
        return self.data["raw"]


class PdfReader(TextReader):
    """imports PDF text; currently using linux only command line *pdftotext*
     utility

    >>> t = PdfReader('testdata/simpletest/textreader.pdf')
    >>> print t.get_text()
    Test Journal Article
    by the authors
    """

    def __init__(self, filename):
        self.data = {"filetype": "pdf",
                     "raw": self.import_pdf(filename)}

    def import_pdf(self, filename):
        """
            runs pdftotext command line util via python subprocess

        """
        rawtext = subprocess.check_output(['pdftotext', filename, '-'])
        return rawtext.strip() # remove any multiple blank lines at the end


class HtmlReader(TextReader):
    """imports HTML text using BeautifulSoup

    >>> t = HtmlReader('testdata/simpletest/textreader.html')
    >>> print t.get_text()
    Test Journal Article
    by the authors

    """
    def __init__(self, filename):
        self.data = {"filetype": "html",
                     "raw": self.import_html(filename)}

    def import_html(self, filename):
        "retrieves text from an HTML file"
        html = self.import_text(filename)
        raw_text = BeautifulSoup(html).body.get_text().strip()
        return raw_text


class JournalReader:
    "Journal reader base class"

    def __init__(self, filename):
        self.file_obj = self.import_file_obj(filename)

    def import_file_obj(self, filename):
        "adds an appropriate reader class to the jread object"
        ext = os.path.splitext(filename)[-1].lower()
        if ext == ".pdf":
            file_obj = pdf_reader(filename)
        elif ext == ".htm" or ext == ".html":
            file_obj = html_reader(filename)
        else:
            file_obj = text_reader(filename)
        return file_obj

    def get_text(self):
        "returns the contents as plain text"
        return self.file_obj.get_text()


def main():
    import doctest
    doctest.testmod()




if __name__ == '__main__':
    main()

