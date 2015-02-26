#
#   drugbank test parser
#

import xml.etree.cElementTree as ET
import progressbar

FILENAME = 'drugbank.xml'



def main():

    output = {}

    print "Parsing 100MB XML - short delay now"

    tree = ET.parse(FILENAME)

    print "done!"

    print "Finding all drugs"

    drugs = tree.findall('{http://drugbank.ca}drug')

    print "Adding synonyms and brand names"

    # p = progressbar.ProgressBar(len(drugs))

    for drug in drugs:
        # p.tap()
        generic_name = drug.find("{http://drugbank.ca}name").text.lower()


        # for brand_name in drug.findall("{http://drugbank.ca}brands/{http://drugbank.ca}brand"): #try without brands
        #     output[brand_name.text.lower()] = generic_name

        for synonym in drug.findall("{http://drugbank.ca}synonyms/{http://drugbank.ca}synonym"):
            output[synonym.text.lower()] = generic_name
        



            
            # path_s = '/'.join(path)
            # if path_s == "{http://drugbank.ca}drugs/{http://drugbank.ca}drug/{http://drugbank.ca}name":
            #     generic = elem.text
            #     output[generic] = generic
            # elif path_s == "{http://drugbank.ca}drugs/{http://drugbank.ca}drug/{http://drugbank.ca}brands/{http://drugbank.ca}brand":
            #     brand = elem.text
            #     # print brand
            #     output[brand] = generic
            # path.pop()


    print "adrenaline is actually %s" % (output["adrenaline"],)

if __name__ == '__main__':
    main()