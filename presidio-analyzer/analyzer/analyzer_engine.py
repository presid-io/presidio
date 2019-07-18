import json
import uuid

import analyze_pb2
import analyze_pb2_grpc
import common_pb2

from analyzer.logger import Logger
from analyzer.app_tracer import AppTracer

DEFAULT_LANGUAGE = "en"
logger = Logger()


class AnalyzerEngine(analyze_pb2_grpc.AnalyzeServiceServicer):

    def __init__(self, registry=None, nlp_engine=None,
                 app_tracer=None, enable_trace_pii=False):
        if not nlp_engine:
            from analyzer.nlp_engine import SpacyNlpEngine
            nlp_engine = SpacyNlpEngine()
        if not registry:
            from analyzer import RecognizerRegistry
            registry = RecognizerRegistry()
        if not app_tracer:
            app_tracer = AppTracer()
        # load nlp module

        self.nlp_engine = nlp_engine
        # prepare registry
        self.registry = registry
        # load all recognizers
        registry.load_predefined_recognizers()
        self.app_tracer = app_tracer
        self.enable_trace_pii = enable_trace_pii

    # pylint: disable=unused-argument
    def Apply(self, request, context):
        """
        GRPC entry point to Presidio-Analyzer
        :param request: Presidio Analyzer resuest of type AnalyzeRequest
        :param context:
        :return: List of [AnalyzeResult]
        """
        logger.info("Starting Analyzer's Apply")

        entities = AnalyzerEngine.__convert_fields_to_entities(
            request.analyzeTemplate.fields)
        language = AnalyzerEngine.get_language_from_request(request)

        # correlation is used to group all traces related to on request

        threshold = request.analyzeTemplate.resultsScoreThreshold
        all_fields = request.analyzeTemplate.allFields

        # A unique identifier for a request, to be returned in the response
        correlation_id = str(uuid.uuid4())
        results = self.analyze(correlation_id, request.text,
                               entities, language,
                               all_fields,
                               threshold)

        # Create Analyze Response Object
        response = analyze_pb2.AnalyzeResponse()

        response.requestId = correlation_id
        # pylint: disable=no-member
        response.analyzeResults.extend(
            AnalyzerEngine.__convert_results_to_proto(results))

        logger.info("Found %d results", len(results))
        return response

    @staticmethod
    def __remove_duplicates(results):
        # bug# 597: Analyzer remove duplicates doesn't handle all cases of one
        # result as a substring of the other
        results = sorted(results,
                         key=lambda x: (-x.score, x.start, x.end - x.start))
        filtered_results = []

        for result in results:
            if result.score == 0:
                continue

            valid_result = True
            if result not in filtered_results:
                for filtered in filtered_results:
                    # If result is equal to or substring of
                    # one of the other results
                    if result.start >= filtered.start \
                            and result.end <= filtered.end:
                        valid_result = False
                        break

            if valid_result:
                filtered_results.append(result)

        return filtered_results

    @staticmethod
    def __remove_low_scores(results, score_threshold):
        new_results = []
        for result in results:
            if result.score >= score_threshold:
                new_results.append(result)

        return new_results

    @classmethod
    def get_language_from_request(cls, request):
        language = request.analyzeTemplate.language
        if language is None or language == "":
            language = DEFAULT_LANGUAGE
        return language

    def analyze(self, correlation_id, text, entities, language, all_fields,
                score_threshold=0.1):
        """
        analyzes the requested text, searching for the given entities
         in the given language
        :param correlation_id: cross call ID for this request
        :param text: the text to analyze
        :param entities: the text to search
        :param language: the language of the text
        :param all_fields: a Flag to return all fields
        of the requested language
        :param score_threshold: A minimum value for which
        to return an identified entity
        :return: an array of the found entities in the text
        """

        recognizers = self.registry.get_recognizers(
            language=language,
            entities=entities,
            all_fields=all_fields)

        if all_fields:
            if entities:
                raise ValueError("Cannot have both all_fields=True "
                                 "and a populated list of entities. "
                                 "Either have all_fields set to True "
                                 "and entities are empty, or all_fields "
                                 "is False and entities is populated")
            # Since all_fields=True, list all entities by iterating
            # over all recognizers
            entities = self.__list_entities(recognizers)

        # run the nlp pipeline over the given text, store the results in
        # a NlpArtifacts instance
        nlp_artifacts = self.nlp_engine.process_text(text, language)

        if self.enable_trace_pii:
            self.app_tracer.trace(correlation_id, "nlp artifacts:"
                                  + nlp_artifacts.to_json())

        results = []
        for recognizer in recognizers:
            # Lazy loading of the relevant recognizers
            if not recognizer.is_loaded:
                recognizer.load()
                recognizer.is_loaded = True

            # analyze using the current recognizer and append the results
            current_results = recognizer.analyze(text, entities, nlp_artifacts)
            if current_results:
                results.extend(current_results)

        self.app_tracer.trace(correlation_id, json.dumps(
            [result.to_json() for result in results]))

        #Remove duplicates or low score results
        results = AnalyzerEngine.__remove_duplicates(results)
        results = AnalyzerEngine.__remove_low_scores(results, score_threshold)

        return results

    @staticmethod
    def __list_entities(recognizers):
        entities = []
        for recognizer in recognizers:
            ents = [entity for entity in recognizer.supported_entities]
            entities.extend(ents)

        return list(set(entities))

    @staticmethod
    def __convert_fields_to_entities(fields):
        # Converts the Field object to the name of the entity
        return [field.name for field in fields]

    @staticmethod
    def __convert_results_to_proto(results):
        proto_results = []
        for result in results:
            res = common_pb2.AnalyzeResult()
            # pylint: disable=no-member
            res.field.name = result.entity_type
            res.score = result.score
            # pylint: disable=no-member
            res.location.start = result.start
            res.location.end = result.end
            res.location.length = result.end - result.start
            proto_results.append(res)

        return proto_results
