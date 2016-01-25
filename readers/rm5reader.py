#
#   rm5reader
#   subclasses xmlbase
#

import xml.etree.cElementTree as ET
import re
import collections
from xmlbase import XMLReader
from pprint import pprint # for testing purposes
from functools import wraps
import pdb
import datetime

def dictzip(*args):
    """
    updates the dicts in l1, a list of dicts, with the 
    corresponding dict from l2
    where l1 and l2 should be the same length
    (and reruns like zip does if they're not)
    """
    for arg in args[1:]:
        [l1item.update(l2item) for l1item, l2item in zip(args[0], arg)]
    return args[0]





def listhandler(list_param_key, default_to_included_studies=True):
    """
    decorates other string handling functions, so
    if input is a string, returns a string
    if input is a list of strings, returns a list of strings
    if input is none, returns values for all included studies

        so:          x -> f(x)
        but  [x, y, z] -> [f(x), f(y), f(z)]
        and       None -> [f(s1), f(s2), f(s3), f(sn), ....]

    """
    def outer_wrapper(func):
        @wraps(func)
        def inner_wrapper(*args, **kwargs):
            # first, get the parameter of interest, which is passed by the decorator as a string
            # or integer.
            # check here whether a string; if so, it's a kwarg
            # if integer, is index of arg
            # also processes defaults
            param_default = args[0].all_ids if default_to_included_studies else None

            if isinstance(list_param_key, int):
                try:
                    list_param_value = args[list_param_key]
                except IndexError:
                    list_param_value = param_default

                if isinstance(list_param_value, list):
                    return {item: func(*(args[:list_param_key] + (item,) + args[list_param_key+1:]), **kwargs) for item in list_param_value}
                elif isinstance(list_param_value, str):
                    return func(*args, **kwargs)
                else:
                    raise TypeError("list parameter %s was %s type; must be either a string, or list of strings" % (list_param_key, type(list_param_value)))

            elif isinstance(list_param_key, str):

                list_param_value = kwargs.get(list_param_key, param_default)

                if isinstance(list_param_value, list):
                    kwargs.pop(list_param_key, None)


                    return {item: func(*args, **dict(kwargs.items()+[(list_param_key, item)])) for item in list_param_value}
                elif isinstance(list_param_value, str):
                    return func(*args, **kwargs)
                else:
                    raise TypeError("list parameter %s was %s type; must be either a string, or list of strings" % (list_param_key, type(list_param_value)))

        return inner_wrapper
    return outer_wrapper


class RM5(XMLReader):

    def __init__(self, filename):
        XMLReader.__init__(self, filename)
        self.map = {}
        self.map["title"] = self.data.find('COVER_SHEET/TITLE')
        self.map["search_date"] = self.data.find('COVER_SHEET/DATES/LAST_SEARCH/DATE')
        self.map["included_studies"] = self.data.find('STUDIES_AND_REFERENCES/STUDIES/INCLUDED_STUDIES')
        self.map["quality"] = self.data.find('QUALITY_ITEMS')
        self.map["characteristics"] = self.data.find('CHARACTERISTICS_OF_STUDIES/CHARACTERISTICS_OF_INCLUDED_STUDIES')
        self.map["analyses"] = self.data.find('ANALYSES_AND_DATA')

        self.all_ids = self.included_study_ids()
        self.comparison_ids = self.get_comparison_ids()

    def camel_case(self, upper_string):
        """
        changes the horrible Cochrane upper case XML to nicer to read version
        """
        return upper_string.replace('_', ' ').title().replace(' ', '')
        
        
    def tree_to_unicode(self, tree, strip_tags=True):
        """
        returns unicode of elementtree contents (in an slightly complicated way!)
        cochrane XML may contain HTML, this allows it to be extracted properly
        """
        if tree is not None:
            if strip_tags:
                return (" ".join(ET.tostringlist(tree, method="text", encoding="utf-8"))).strip()
            else:
                return ET.tostring(tree, method="text", encoding="utf-8")
        else:
            return None

    @listhandler(1, default_to_included_studies=False)
    def tree_search(self, query, tree):
        if tree:
            return self.tree_to_unicode(tree.find(query))
        else:
            return None

    def title(self):
        return self.tree_to_unicode(self.map["title"])

    def review_group(self):
        """
        returns the unique ID of the group which authored the review
        """

        return self.data.getroot().attrib.get("GROUP_ID")
        
    def cdno(self):
        """
        first tries to get cdno from filename (most reliable), else try to extract from DOI
        """
        try:
            cdno = re.search('(?:CD|MR)[0-9]+', self.filename).group(0)
        except:
            doi = self.data.getroot().attrib.get("DOI")
            parts = doi.split('.')
            for part in parts:
                if part[:2] == "CD" or part[:2] == "MR":
                    cdno = part
                else:
                    cdno = "Unknown"
        return cdno

    def review_type(self):
        """
        is this review looking at an intervention, diagnostic tests, methodology, or an overview or reviews
        """
        return self.data.getroot().attrib.get("TYPE")

    def search_date(self):
        """
        return a date object with the most recent search date
        """

        # print self.map["search_date"]
        day, month, year = (int(self.map["search_date"].attrib.get(id_str)) for id_str in ["DAY", "MONTH", "YEAR"])

        return datetime.date(day=day, month=month, year=year)





    def included_study_ids(self):
        """
        returns the Cochrane review internal unique id for all included studies
        (note that the id is unique in the *review* only, and not across the library)
        """
        try:
            return [study.attrib.get("ID") for study in self.map["included_studies"].findall("STUDY")]
        except AttributeError:
            return [] # where there are no included studies

    def get_comparison_ids(self):
        """
        returns the Cochrane review internal unique id for all included comparisons
        (note that the id is unique in the *review* only, and not across the library)
        """
        try:
            return [study.attrib.get("ID") for study in self.map["analyses"].findall("COMPARISON")]
        except AttributeError:
            return []




    @listhandler(1)
    def tree_attributes(self, study_id, tree):
        return tree.attrib.get(study_id)

    @listhandler("study_id")
    def references(self, study_id=None, hide_secondary_refs=False):
        """
        By default, returns details of primary study, or first study if none
        listed as primary
        if hide_secondary_refs is set to False, then
        returns a list of *all* citations for each study
        """

        search_string = ".//STUDY[@ID='%s']" % study_id
        study = self.map["included_studies"].find(search_string)
       
        cites = study.findall("REFERENCE")
        part_ids = ["TI", "SO", "AU", "YR", "PG", "VL", "NO"]

        if hide_secondary_refs:

            # loop through all cites
            for cite in cites:
                if self.tree_attributes("PRIMARY", cite):
                    # if primary: found it!
                    break
            else:
                # no primary found
                cite = cites[0]
            part_contents = self.tree_search(part_ids, cite)

        else:
            part_contents = [self.tree_search(part_ids, cite) for cite in cites]
        return part_contents

    
    @listhandler("study_id")
    def quality(self, study_id=None):
        """
        Returns risk of bias information about a study id
        """

        quality_items = self.map["quality"].findall("QUALITY_ITEM")

        output = []

        for item in quality_items:

            domain_name = self.tree_search("NAME", item)
            domain_description = self.tree_search("DESCRIPTION", item)


            level = self.tree_attributes("LEVEL", item)
            group_data = {}

            if level == "GROUP":
                groups = item.findall("QUALITY_ITEM_DATA_ENTRY_GROUP")

                for group in groups:

                    group_key = self.tree_attributes("ID", group)
                    group_value = self.tree_search("NAME", group)
                    group_data[group_key] = group_value


            data = item.find("QUALITY_ITEM_DATA")

            search_string = "QUALITY_ITEM_DATA_ENTRY[@STUDY_ID='%s']" % study_id
            data_items = data.findall(search_string)

            for data_item in data_items:
                if level == "GROUP":
                    rating_scope = group_data[self.tree_attributes("GROUP_ID", data_item)]
                else:
                    rating_scope = "STUDY_LEVEL"

                description = self.tree_search("DESCRIPTION", data_item)
                rating = self.tree_attributes("RESULT", data_item)
                

                output.append({"DOMAIN": domain_name, "DOMAIN_DESCRIPTION": domain_description, 
                                   "RATING": rating, "SCOPE": rating_scope, "DESCRIPTION": description})

        return output


    @listhandler("study_id")
    def char_refs(self, study_id=None):
        """
        return characteristics of an included study
        """

        search_string = "INCLUDED_CHAR[@STUDY_ID='%s']" % study_id
        study = self.map["characteristics"].find(search_string)

        characteristics = {"CHAR_METHODS": self.tree_search("CHAR_METHODS", study),
                           "CHAR_PARTICIPANTS": self.tree_search("CHAR_PARTICIPANTS", study),
                           "CHAR_INTERVENTIONS": self.tree_search("CHAR_INTERVENTIONS", study),
                           "CHAR_OUTCOMES": self.tree_search("CHAR_OUTCOMES", study),
                           "CHAR_NOTES": self.tree_search("CHAR_NOTES", study)}

        return characteristics



    @listhandler("study_id")
    def full_parse(self, study_id=None):

        parse = {"CHARACTERISTICS": self.char_refs(study_id=study_id),
                 "QUALITY": self.quality(study_id=study_id),
                 "REFERENCE": self.references(study_id=study_id),
                 "GROUP_ID": self.review_group()}
        

        return parse

    def ref_characteristics(self):

        studies_characteristics = self.data.findall("CHARACTERISTICS_OF_STUDIES/CHARACTERISTICS_OF_INCLUDED_STUDIES/INCLUDED_CHAR")
        output = {}
        
        for study_characteristics in studies_characteristics:
            
            study_id = study_characteristics.attrib.get("STUDY_ID")
            
            characteristics = {"CHAR_METHODS": self._ETfind("CHAR_METHODS", study_characteristics),
                               "CHAR_PARTICIPANTS": self._ETfind("CHAR_PARTICIPANTS", study_characteristics),
                               "CHAR_INTERVENTIONS": self._ETfind("CHAR_INTERVENTIONS", study_characteristics),
                               "CHAR_OUTCOMES": self._ETfind("CHAR_OUTCOMES", study_characteristics),
                               "CHAR_NOTES": self._ETfind("CHAR_NOTES", study_characteristics)}
            
            output.update({study_id: characteristics})
            
        return output
            
            
        


    def sof_table(self):
        return self._ETfind("SOF_TABLES/SOF_TABLE", self.data, strip_tags=False)

    def dich_outcomes(self):

        output = []

        try:
            comp_data_entries = self.map["analyses"].findall("COMPARISON")
        except:
            return []

        for comp_data_entry in comp_data_entries:

            

            comp_name = self.tree_search("NAME", comp_data_entry)
            comp_no = comp_data_entry.attrib.get("NO")

            dich_data_entries = comp_data_entry.findall("DICH_OUTCOME")

        
            for dich_data_entry in dich_data_entries:

                variables_of_interest = ["NAME", "GROUP_LABEL_1", "GROUP_LABEL_2", "GRAPH_LABEL_1", "GRAPH_LABEL_2"]

                dich_data_dict = {self.camel_case(id_str): self.tree_search(id_str, dich_data_entry) for id_str in variables_of_interest}

                dich_data_dict["CmpNo"] = comp_no
                dich_data_dict["CmpName"] = comp_name

                variables_of_interest = ["TOTAL_1", "TOTAL_2", "CI_START", "CI_END", "EVENTS_1", "EVENTS_2", "EFFECT_SIZE", "ESTIMABLE", "STUDIES", "NO", "EFFECT_MEASURE"]

                # dich_data_dict["AnalysisNo"] = dich_data_entry.attrib.get("NO")

                dich_data_dict.update({self.camel_case(id_str): dich_data_entry.attrib.get(id_str) for id_str in variables_of_interest})

                # now get the individual studies

                study_entries = dich_data_entry.findall("DICH_DATA")

                dich_data_dict["StudyData"] = []

                variables_of_interest = ["TOTAL_1", "TOTAL_2", "CI_START", "CI_END", "EVENTS_1", "EVENTS_2", "EFFECT_SIZE", "ESTIMABLE", "STUDIES", "NAME", "STUDY_ID"]

                for study_entry in study_entries:
                    dich_data_dict["StudyData"].append({self.camel_case(id_str): study_entry.attrib.get(id_str) for id_str in variables_of_interest})





                


                output.append(dich_data_dict)
        return output

        



        





def main():

    # example - show some data from a random review
    # and give some details about the first included trial in the review
    
    import random
    import glob

    rm5_files_path = '/Users/iain/data/cdsr/2013/'
    rm5_files = glob.glob(rm5_files_path + '*.rm5')

    # reader = RM5(random.choice(rm5_files))
    reader = RM5(rm5_files_path + "CD004659 v. 5.0 Antiplatelet agents for preventing pre-eclampsia and its complications.rm5")
    # reader = RM5(rm5_files[1020])


    print "Title:"
    print reader.title()
    print

    # print "Review type:"
    # print reader.review_type()
    # print

    print "Included study IDs:"
    print reader.included_study_ids()
    print

    # print "Excluded study IDs:"
    # print reader.excluded_study_ids()
    # print

    ids = reader.included_study_ids()

    print
    print "Primary reference details for %s" % (ids[1])
    print reader.references()
    print

    # print "All citation details for %s" % (ids[0])
    # print [len(refs) for refs in reader.references(hide_secondary_refs=False)]
    # print reader.references(study_ids=None, hide_secondary_refs=False)


    print "Cochrane ID:"
    print reader.cdno()
    print

    print "Quality info"
    print reader.quality(study_id=ids[1])



    refs = reader.references() # False just retrieves the citation
    print "No included studies: %d" % (len(refs),)
    print



    print "First included study title:"
    print refs[ids[0]][0]["TI"]
    print

    print "Population details"
    print reader.char_refs(study_id=ids[0])["CHAR_PARTICIPANTS"]
    print

    print "Written by the following review group:"
    print reader.review_group()

    # print "Risk of bias"
    # print

    # for i, domain in enumerate(refs[ids[0]]["QUALITY"]):
    #     print "Domain number %d" % (i, )
    #     print "Name\t\t" + domain["DOMAIN"]
    #     print "Description\t" + domain["DESCRIPTION"]
    #     print "Rating\t\t" + domain["RATING"]
        
    print reader.references().values()
    print
    # print reader.references(study_id = ['STD-Thomas-1987', 'STD-Vercellini-1999'])

    # print
    # print reader.sof_table()


    # print reader.quality(study_id='STD-Brazil-1992a')

    # print "Analyses"
    # print
    # print 
    # analyses = reader.dich_outcomes()

    # # print analyses

    # # for analysis in analyses:
    #     # print analysis["GroupLabel2"]
        
    #     if any(control_name in analysis["GroupLabel2"].lower() for control_name in ["placebo", "control", "usual care"]):
    #         print analysis["Name"]
    

def _refs_PG(ref):
    "Accepts string with page number e.g. 1254-63; outputs tuple (start, end)"
    
    if ref is None:
        return ("", "")
    
    ref_parts = ref.split('-')

    start = ref_parts[0]
    
    if len(ref_parts) == 1:
        return (start, start)

    elif len(ref_parts) == 2:
        end = ref_parts[1]
        
        if len(end) < len(start):
            end = start[:(len(start)-len(end))] + end
        
        return (start, end)
    else:
        return ("", "")
        
def _refs_AU(author_string):
    "Accepts string with list of authors; outputs list, et als removed"
    author_string = re.sub("[\s\.,]*et al", "", author_string)
    authors = author_string.split(', ')
    return authors


if __name__ == '__main__':
    main()






