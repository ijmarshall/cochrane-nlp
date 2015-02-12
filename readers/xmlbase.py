#
# XML reader base class
#

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

class XMLReader():
    """
    new faster version which doesn't use NLTK
    returns plain text - tokenization should be done externally
    """

    def __init__(self, filename=None):
        self.filename = filename
        self.data = self.load(filename)
        self.section_map = {}
                    
    def load(self, filename):
        return ET.parse(filename)
            
    def _ET2unicode(self, ET_instance, strip_tags=True):
        "returns unicode of elementtree contents"
        if ET_instance is not None:
            if strip_tags:
                # print "tags stripped!"
                return (" ".join(ET.tostringlist(ET_instance, method="text", encoding="utf-8"))).strip()
            else:
                return ET.tostring(ET_instance, method="xml", encoding="utf-8")
        else:
            return ""

    def _ETfind(self, element_name, ET_instance, strip_tags=True):
        "finds (first) subelement, returns unicode of contents if present, else returns None"
        subelement = ET_instance.find(element_name)
        if subelement is not None:
            return self._ET2unicode(subelement, strip_tags=strip_tags)
        else:
            return ""

    def text_filtered(self, part_id=None):
        if type(part_id) is str:
            return self._ET2unicode(self.xml_filtered(part_id=part_id))
        elif type(part_id) is list:
            return {p: self._ET2unicode(self.xml_filtered(part_id=p)) for p in part_id}

    def text_filtered_all(self, part_id=None):
        if type(part_id) is str:
            return [self._ET2unicode(part) for part in self.xml_filtered_all(part_id=part_id)]
        elif type(part_id) is list:
            return {p: [self._ET2unicode(part) for part in self.xml_filtered_all(part_id=p)] for p in part_id}

    def text_all(self):
        output = {}
        for part_id, loc in self.section_map.iteritems():
            output[part_id] = self._ET2unicode(self.data.find(loc))
        return output

    def xml_filtered_all(self, part_id=None):        
        return self.data.findall(self.section_map[part_id])

    def xml_filtered(self, part_id=None):        
        return self.data.find(self.section_map[part_id])
        