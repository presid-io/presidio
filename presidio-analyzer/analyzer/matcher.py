import logging
import regex as re
import en_core_web_sm
import common_pb2
import template_pb2
from field_types import field_type, field_factory
from field_types.globally import ner

CONTEXT_SIMILARITY_THRESHOLD = 0.65
NER_PROBABILITY = 0.8

class Matcher(object):
    def __init__(self):
        """Constructor
        Load spacy model once
        """

        self.nlp = en_core_web_sm.load()

    def __calculate_context_similarity(self, context, field):
        '''context similarity = max similarity between context token and a field keyword'''
        lemmatized_context = list(map(lambda t: t.lemma_, self.nlp(context.lower())))
        max_similarity = 0.0

        for context_token in lemmatized_context:
            for keyword in field.context:
                similarity = self.nlp(context_token).similarity(self.nlp(keyword)) 
                if similarity >= CONTEXT_SIMILARITY_THRESHOLD:
                    max_similarity = max(max_similarity, similarity)

        return min(max_similarity, 1)

    def __calculate_probability(self, doc, context, field, start):
        if field.should_check_checksum:
            if field.check_checksum() is not True:
                logging.info("Checksum failed for " + field.text)
                return 0
            else:
                return 1.0

        probability = 0.5
        factor = 0.5

        '''calculate probability based on context '''
        context_similarity = self.__calculate_context_similarity(context, field)
        if context_similarity >= CONTEXT_SIMILARITY_THRESHOLD:
            probability += context_similarity * factor

        return min(probability, 1)
        
    def __create_result(self, doc, context, current_field, start, end):

        res = common_pb2.AnalyzeResult()
        res.field.name = current_field.name
        res.text = current_field.text

        # Validate checksum
        res.probability = self.__calculate_probability(
                doc, context, current_field, start)

        res.location.start = start
        res.location.end = end
        res.location.length = end - start

        logging.info(
            f"field: '{res.field}' Value: '{res.text}' Span: '{start}:{end}' Probability: '{res.probability}'"
        )
        return res

    def __get_context(self, doc, start, end):
        return doc.text[0:start] + doc.text[end + 1:]

    def __check_pattern(self, doc, results, field):
        for _, check_type_value in field.regexes.items():
            matches = re.finditer(
                    check_type_value,
                    doc.text,
                    flags=re.IGNORECASE | re.DOTALL | re.MULTILINE,
                    overlapped=False,
                    partial=False,
                    concurrent=True)
            
            for match in matches:
                start, end = match.span()
                field.text = doc.text[start:end]
                context = self.__get_context(doc, start, end)

                # Skip empty results
                if field.text == '':
                    continue

                # Don't add duplicate
                if len(field.regexes) > 1 and any(
                        ((x.location.start == start) or (
                            x.location.end == end)) and
                        ((
                            x.field.name == field.name))
                        for x in results):
                    continue

                res = self.__create_result(doc, context, field, start, end)
                if res is None or res.probability == 0:
                    continue

                # Don't add overlap
                # if any(x.location.end >= start and x.probability == 1.0
                #        for x in results):
                #     continue

                results.append(res)

    def __match_ner(self, label, field_type_filter):
        if field_type_filter == "LOCATION" and (label == 'GPE'
                                                or label == 'LOC'):
            return True

        if field_type_filter == "PERSON" and label == 'PERSON':
            return True

        if field_type_filter == "DATE_TIME" and (label == 'DATE'
                                                 or label == 'TIME'):
            return True

        if field_type_filter == "NRP" and label == 'NORP':
            return True

        return False

    def __check_ner(self, doc, results, current_field):
        for ent in doc.ents:
            if self.__match_ner(ent.label_, current_field.name) is False:
                continue
            current_field.text = ent.text
            context = self.__get_context(doc, ent.start_char, ent.end_char)
            res = self.__create_result(doc, context, current_field, ent.start_char,
                                       ent.end_char)
            res.probability = NER_PROBABILITY

            if res is not None:
                results.append(res)

        return results

    def __sanitize_text(self, text):
        # text = text.replace('\n', ' ')
        # text = text.replace('\r', ' ')
        return text

    def analyze_text(self, text, field_type_filters):
        """Analyze text.

        Args:
            text: text to analyzer.
            field_type_filters: filters array such as ["PERSON", "LOCATION"]
        """

        results = []
        field_type_string_filters = []

        if field_type_filters is None or not field_type_filters:
            field_type_string_filters = field_factory.types_refs
        else:
            for field_type in field_type_filters:
                field_type_string_filters.append(field_type.name)

        sanitized_text = self.__sanitize_text(text)
        doc = self.nlp(sanitized_text)
        for field_type_string_filter in field_type_string_filters:
            current_field = field_factory.FieldFactory.create(
                field_type_string_filter)

            # Check for ner field
            if isinstance(current_field, type(ner.Ner())):
                current_field.name = field_type_string_filter
                self.__check_ner(doc, results, current_field)
            else:
                self.__check_pattern(doc, results, current_field)

        results.sort(key=lambda x: x.location.start, reverse=False)
        return results
