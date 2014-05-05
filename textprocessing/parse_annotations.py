
# annotation parser that processes our hashtags

import re

def get_abstracts(filename):

    output = []    
    abstract_buffer = []
    notes_buffer = []
    last_abstract_no = 0
    last_pmid = 0
    last_biviewid = 0
    
    with open(filename, 'rb') as f:

        for line in f: # fast forward to abstract 1
            m = re.match("Abstract 1 of [1-9][0-9]*", line)
            if m:
                record_abstract=True
                last_abstract_no = 1
                break

        for line in f:

            m = re.match("Abstract ([1-9][0-9]*) of [1-9][0-9]*", line.strip())
            if m:
                record_abstract = True
                output.append({"abstract": "\n".join(abstract_buffer),
                               "notes": notes_buffer,
                               "pmid": last_pmid,
                               "biviewid": last_biviewid,
                               "annotid": last_abstract_no})
                abstract_buffer, notes_buffer = [], []
                last_abstract_no = int(m.group(1))

                continue

            m = re.match("BiviewID ([0-9]+); PMID ([0-9]+)", line)
            if m:
                record_abstract = False
                last_biviewid = int(m.group(1))
                last_pmid = int(m.group(2))
                continue

            if line.strip():

                if record_abstract:
                    abstract_buffer.append(line)
                else:
                    notes_buffer.append(line)
        else:
            output.append({"abstract": "\n".join(abstract_buffer),
               "notes": notes_buffer})


    return output


















def main():
    a = get_abstracts("data/drug_trials_in_cochrane_BCW.txt")
    b = get_abstracts("data/drug_trials_in_cochrane_IJM.txt")

    

    i = 128

    print
    print a[i]
    print
    print b[i]



if __name__ == '__main__':
    main()