

# I had to apply this patch: https://code.google.com/p/banyan/issues/detail?id=5
from banyan import *
from abc import ABCMeta, abstractmethod

class Pipeline(object):
    __metaclass__ = ABCMeta

    pipeline_title = ""

    def __preprocess(self, pages):
        # we need to do two things, create a single string for each page
        # and establish a interval-tree to figure out the original nodes
        parsed = []
        for idx, page in enumerate(pages):
            if page is not None:
                textNodes = [node["str"] for node in page]

                total = 0
                ranges = []
                for txt in textNodes:

                    start = total
                    total += len(txt)

                    ranges.append((start, total))

                    total += 1  # we're adding an extra space
                interval_tree = SortedSet(ranges, key_type = (int, int), updator = OverlappingIntervalsUpdator)
                page_str = " ".join(textNodes)

                parsed.append({"str": page_str,
                               "length": total,
                               "intervals": interval_tree,
                               "ranges": ranges})
            else:
                parsed.append({})

        return parsed

    def get_page_offsets(self, page_lengths):
        " takes list of page lengths, returns cumulative list for offsets "
        # we store this because we want per document, not per page
        def accumulate(x, l=[0]):
            # since list l held by reference, value is stored between function calls!
            l[0] += x
            return l[0]
        return map(accumulate, [0] + page_lengths)

    def __postprocess_annotations(self, parsed_input, predictions):
        # get the page lengths, and the page offsets in the whole doc string
        page_lengths = [page["length"] for page in parsed_input]
        total_length = self.get_page_offsets(page_lengths)

        # Now we need to get /back/ to the page and node indexes
        annotations = []
        for sentence_bound, labels in predictions.iteritems():
            page_nr = next(i for i,v in enumerate(total_length) if v >= sentence_bound[0]) - 1
            page = parsed_input[page_nr]
            offset = total_length[page_nr]

            bound = (sentence_bound[0] - offset, sentence_bound[1] - offset)

            nodes = [page["ranges"].index(x) for x in page["intervals"].overlap(bound)]

            annotations.append({
                "page": page_nr,
                "nodes": nodes,
                "labels": labels})

        return annotations

    def run(self, input):
        parsed_pages = self.__preprocess(input)

        # get the predictions
        full_text = ' '.join(page["str"] for page in parsed_pages)

        document_predictions, sentence_predictions = self.predict(full_text)

        annotations = self.__postprocess_annotations(parsed_pages, sentence_predictions)

        return {
            "title": self.pipeline_title,
            "document": document_predictions,
            "annotations": annotations}
