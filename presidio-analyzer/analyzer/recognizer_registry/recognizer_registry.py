import time
import logging

from analyzer.recognizer_registry import RecognizerStoreApi
from analyzer.predefined_recognizers import CreditCardRecognizer, \
    SpacyRecognizer, CryptoRecognizer, DomainRecognizer, \
    EmailRecognizer, IbanRecognizer, IpRecognizer, NhsRecognizer, \
    UsBankRecognizer, UsLicenseRecognizer, \
    UsItinRecognizer, UsPassportRecognizer, UsPhoneRecognizer, \
    UsSsnRecognizer


class RecognizerRegistry:
    """
    Detects, registers and holds all recognizers to be used by the analyzer
    """

    def __init__(self, recognizer_store_api=RecognizerStoreApi(),
                 recognizers=None):
        """
        :param recognizer_store_api: An instance of a class that has custom
               recognizers management functionallity (insert, update, get,
               delete). The default store if nothing is else is provided is
               a store that uses a persistent storage
        :param recognizers: An optional list of recognizers that will be
               available in addition to the predefined recognizers and the
               custom recognizers
        """
        if recognizers:
            self.recognizers = recognizers
        else:
            self.recognizers = []
        self.loaded_timestamp = None
        self.loaded_custom_recognizers = []
        self.store_api = recognizer_store_api

    def load_predefined_recognizers(self):
        #   TODO: Change the code to dynamic loading -
        # Task #598:  Support loading of the pre-defined recognizers
        # from the given path.
        # Currently this is not integrated into the init method to speed up
        # loading time if these are not actually needed (SpaCy for example) is
        # time consuming to load
        self.recognizers.extend([
            CreditCardRecognizer(),
            SpacyRecognizer(),
            CryptoRecognizer(), DomainRecognizer(),
            EmailRecognizer(), IbanRecognizer(),
            IpRecognizer(), NhsRecognizer(),
            UsBankRecognizer(), UsLicenseRecognizer(),
            UsItinRecognizer(), UsPassportRecognizer(),
            UsPhoneRecognizer(), UsSsnRecognizer()])

    def get_recognizers(self, entities=None, language=None):
        """
        Returns a list of recognizers, which support the specified name and
        language. if no language and entities are given, all the available
        recognizers will be returned
        :param entities: the requested entities
        :param language: the requested language
        :return: A list of recognizers which support the supplied entities
        and language
        """

        if language is None and entities is None:
            return self.recognizers

        if language is None:
            raise ValueError("No language provided")

        if entities is None:
            raise ValueError("No entities provided")

        all_possible_recognizers = self.recognizers.copy()
        custom_recognizers = self.get_custom_recognizers()
        all_possible_recognizers.extend(custom_recognizers)
        logging.info("Found %d (total) custom recognizers",
                     len(custom_recognizers))

        # filter out unwanted recognizers
        to_return = []
        for entity in entities:
            subset = [rec for rec in all_possible_recognizers if
                      entity in rec.supported_entities
                      and language == rec.supported_language]

            if not subset:
                logging.warning(
                    "Entity " + entity +
                    " doesn't have the corresponding recognizer in language :"
                    + language)
            else:
                to_return.extend(subset)

        logging.info(
            "Returning a total of %d recognizers (predefined + custom)",
            len(to_return))

        if not to_return:
            raise ValueError(
                "No matching recognizers were found to serve the request.")

        return to_return

    def get_custom_recognizers(self):
        """
        Returns a list of custom recognizers retrieved from the store object
        """

        if self.loaded_timestamp is not None:
            logging.info(
                "Analyzer loaded custom recognizers on: %s (%s)",
                time.strftime(
                    '%Y-%m-%d %H:%M:%S',
                    time.localtime(int(self.loaded_timestamp))),
                self.loaded_timestamp)
        else:
            logging.info("Analyzer loaded custom recognizers on: Never")

        lst_update = self.store_api.get_latest_timestamp()
        # is update time is not set, no custom recognizers in storage, skip
        if lst_update > 0:
            logging.info(
                "Persistent storage was last updated on: %s (%s)",
                time.strftime('%Y-%m-%d %H:%M:%S',
                              time.localtime(lst_update)), lst_update)
            # check if anything updated since last time
            if self.loaded_timestamp is None or \
                    lst_update > self.loaded_timestamp:
                self.loaded_timestamp = int(time.time())

                self.loaded_custom_recognizers = []
                # read all values
                logging.info(
                    "Requesting custom recognizers from the storage...")

                raw_recognizers = self.store_api.get_all_recognizers()
                if raw_recognizers is None or not raw_recognizers:
                    logging.info(
                        "No custom recognizers found")
                    return []

                logging.info(
                    "Found %d recognizers in the storage",
                    len(raw_recognizers))
                self.loaded_custom_recognizers = raw_recognizers

        return self.loaded_custom_recognizers
