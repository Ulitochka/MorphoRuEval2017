import unittest

from Tagger.Morpho_tagger import MorphoTagger
from tools.workers.Config_worker import Config


class TestDataPreparator(unittest.TestCase):
    def setUp(self):
        config = Config(config_string='{"CORPORA": "opencorpora", "GRAMMAR_CLASSES_POSITIONS": {"Animacy": 1, "Aspect": 2, "Case": 3, "Degree": 4, '
                                      '"Gender": 5, "Mood": 6, "Number": 7, "Person": 8, "Tense": 9, "Variant": 10, '
                                      '"VerbForm": 11, "Voice": 12, "POS": 0},'
                                      '"FILES_NAMES": ["test"], '
                                      '"FEATURES_OPTIONS": {"ALL_TOKENS_IN_CONTEXT": 0, "CONTEXT_RIGHT": 2, '
                                      '"CONTEXT_LEFT": 2, "SUFFIX_OPTIONS": [-2, -3], "ISUPPER": 1, "ISTITLE": 1, '
                                      '"ISDIGIT": 1}, '
                                      '"LABELS_OPTIONS": {"POS": [1, 3], "POS&GRAMMAR": [1, 3, 5], "POSvsGRAMMAR": [3, 5], '
                                      '"SUB_TAG": "sub_tag_from_file"}, "FEATURES_OPTIONS": {}, "BINARY": "False", '
                                      '"FORMAT": "dict", "DIM_RED": "False"}')
        self.test_tagger = MorphoTagger(config, model_name='crf_', text=True, eval_type=True)

    def test_merge_predictions_beta(self):
        predicted_data = {
            'POS': [
                ['ADP', 'ADJ', 'NOUN', 'NOUN', 'NOUN', 'PROPN', 'PUNCT', 'NOUN', 'ADP', 'PROPN', 'VERB', 'ADP', 'NUM',
                 'NOUN', 'PUNCT'],
                ['VERB', 'ADP', 'NOUN', 'ADP', 'ADJ', 'NOUN', 'NOUN', 'ADJ', 'CONJ', 'ADJ', 'NOUN', 'VERB', 'NOUN',
                 'CONJ', 'NOUN', 'NOUN', 'PUNCT']
            ],
            'Gender': [
                ['O', 'Masc', 'Masc', 'O', 'O', 'O', 'O', 'Masc', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
                ['Masc', 'O', 'O', 'O', 'Masc', 'Masc', 'O', 'Masc', 'O', 'Masc', 'Masc', 'Masc', 'Masc', 'O', 'O',
                 'O', 'O']
            ],
            'Number': [
                ['O', 'Sing', 'Sing', 'Sing', 'Sing', 'Sing', 'O', 'O', 'O', 'Sing', 'Sing', 'O', 'O', 'O', 'O'],
                ['Sing', 'O', 'Sing', 'O', 'Sing', 'Sing', 'Sing', 'Sing', 'O', 'Sing', 'Sing', 'Sing', 'Sing', 'O',
                 'Sing', 'Sing', 'O']
            ]
        }
        true_result = [
            ['ADP', 'ADJ_Gender=Masc_Number=Sing', 'NOUN_Gender=Masc_Number=Sing', 'NOUN_Number=Sing',
             'NOUN_Number=Sing', 'PROPN_Number=Sing', 'PUNCT', 'NOUN_Gender=Masc', 'ADP', 'PROPN_Number=Sing',
             'VERB_Number=Sing', 'ADP', 'NUM', 'NOUN', 'PUNCT'],
            ['VERB_Gender=Masc_Number=Sing', 'ADP', 'NOUN_Number=Sing', 'ADP', 'ADJ_Gender=Masc_Number=Sing',
             'NOUN_Gender=Masc_Number=Sing', 'NOUN_Number=Sing', 'ADJ_Gender=Masc_Number=Sing', 'CONJ',
             'ADJ_Gender=Masc_Number=Sing', 'NOUN_Gender=Masc_Number=Sing', 'VERB_Gender=Masc_Number=Sing',
             'NOUN_Gender=Masc_Number=Sing', 'CONJ', 'NOUN_Number=Sing', 'NOUN_Number=Sing', 'PUNCT']
        ]
        fact_result = self.test_tagger.merge_predictions(predicted_data)
        self.assertEqual(fact_result, true_result)

if __name__ == '__main__':
    unittest.main(verbosity=2)