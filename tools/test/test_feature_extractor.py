import unittest
import pprint
from collections import OrderedDict

from tools.workers.Feature_extractor import FeatureExtractor
from tools.workers.Config_worker import Config

import pandas
from pandas.util.testing import assert_frame_equal
from scipy import sparse


class TestFeatureExtractor(unittest.TestCase):
    def setUp(self):
        config = Config(config_string='{"FEATURES_OPTIONS": {"CONTEXT_RIGHT": 2, '
                                                             '"CONTEXT_LEFT": 2, '
                                                             '"SUFFIX_OPTIONS": [-2, -3], '
                                                             '"PREFIX_OPTIONS": 0, '
                                                             '"LOWER_TOKEN": 1,'
                                                             '"ISUPPER": 1, '
                                                             '"ISTITLE": 1, '
                                                             '"ISDIGIT": 1, '
                                                             '"W2V": 0, '
                                                             '"W2V_VECTOR_DIM": 3, '
                                                             '"W2V_CONTEXT": 0, '
                                                             '"W2V_FILE": "ru.bin", '
                                                             '"W2V_PRECISION": 4,'
                                                             '"ISUPPER_CONTEXT": 1, '
                                                             '"ISTITLE_CONTEXT": 1, '
                                                             '"ISDIGIT_CONTEXT": 1, '
                                                             '"SUFFIX_OPTIONS_CONTEXT": 0, '
                                                             '"PREFIX_OPTIONS_CONTEXT": 0, '
                                                             '"LOWER_TOKEN_CONTEXT": 1}}')

        config_with_prefix = Config(config_string='{"FEATURES_OPTIONS": {"CONTEXT_RIGHT": 2, '
                                                                         '"CONTEXT_LEFT": 2, '
                                                                         '"SUFFIX_OPTIONS": [-2, -3], '
                                                                         '"PREFIX_OPTIONS": [2, 3], '
                                                                         '"LOWER_TOKEN": 1,'
                                                                         '"ISUPPER": 1, '
                                                                         '"ISTITLE": 1, '
                                                                         '"ISDIGIT": 1, '
                                                                         '"W2V": 0, '
                                                                         '"W2V_VECTOR_DIM": 0, '
                                                                         '"W2V_CONTEXT": 0, '
                                                                         '"W2V_FILE": 0, '
                                                                         '"W2V_PRECISION": 0,'
                                                                         '"ISUPPER_CONTEXT": 1, '
                                                                         '"ISTITLE_CONTEXT": 1, '
                                                                         '"ISDIGIT_CONTEXT": 1, '
                                                                         '"SUFFIX_OPTIONS_CONTEXT": [-2, -3], '
                                                                         '"PREFIX_OPTIONS_CONTEXT": [2, 3], '
                                                                         '"LOWER_TOKEN_CONTEXT": 1}}')

        config_with_pos = Config(config_string='{"ALGORITHM": "CRF",'
                                               '"CORPORA": "gycrya",'
                                               '"FEATURES_OPTIONS": {"CONTEXT_RIGHT": 1, '
                                                                     '"POS_TAGGER": 1,'
                                                                     '"POS_TAGGER_CONTEXT": 1,'
                                                                     '"CONTEXT_LEFT": 1, '
                                                                     '"SUFFIX_OPTIONS": 0, '
                                                                     '"PREFIX_OPTIONS": 0, '
                                                                     '"LOWER_TOKEN": 1,'
                                                                     '"W2V_CONTEXT": 0, '
                                                                     '"LOWER_TOKEN_CONTEXT": 1}}')

        self.features = FeatureExtractor(config=config, data_type='csv')

        self.features_with_prefix = FeatureExtractor(config=config_with_prefix, data_type='dict')

        self.features_with_pos_tagger = FeatureExtractor(config=config_with_pos, data_type='csv')

        self.test_data = [
            [
                ('В', 'ADP'),
                ('начальный', 'ADJ_Inan_Acc_Pos_Masc_Sing'),
                ('либерализации', 'NOUN_Inan_Gen_Fem_Sing'),
                ('.', 'PUNCT')
            ]
        ]

        self.test_data_w2v_empty = [
            [
                ('Киркоров', 'ADP')
            ]
        ]

    def test_sentence2features_with_pos(self):
        true_result = [
            OrderedDict([('bias', 1.0), ('word.lower()', 'в'), ('+1:word.lower()', 'начальный'),
                         ('-1:word.lower()', 'False'), ('BOS', 'True'), ('EOS', 'False'), ('word.pos_tag()', 'ADP')]),
            OrderedDict([('bias', 1.0), ('word.lower()', 'начальный'), ('-1:word.lower()', 'в'),
                         ('+1:word.lower()', 'либерализации'), ('BOS', 'False'), ('EOS', 'False'), ('word.pos_tag()', 'X')]),
            OrderedDict([('bias', 1.0), ('word.lower()', 'либерализации'), ('-1:word.lower()', 'начальный'),
                         ('+1:word.lower()', '.'), ('BOS', 'False'), ('EOS', 'False'), ('word.pos_tag()', 'NOUN')]),
            OrderedDict([('bias', 1.0), ('word.lower()', '.'), ('-1:word.lower()', 'либерализации'),
                         ('+1:word.lower()', 'False'), ('BOS', 'False'), ('EOS', 'True'), ('word.pos_tag()', 'NOUN')])
        ]
        true_result = pandas.DataFrame(true_result)
        fact_result = self.features_with_pos_tagger.sentence2features(self.test_data)
        assert_frame_equal(fact_result.reset_index(drop=True), true_result.reset_index(drop=True))

    def test_sentence2features(self):
        true_result = [
            OrderedDict([('bias', 1.0), ('word[-2]', 'False'), ('word[-3]', 'False'), ('word.is_upper()', 'True'),
                         ('word.is_title()', 'True'), ('word.is_digit()', 'False'), ('word.lower()', 'в'),
                         ('+1:word.is_upper()', 'False'), ('+1:word.is_title()', 'False'),
                         ('+1:word.is_digit()', 'False'), ('+1:word.lower()', 'начальный'),
                         ('+2:word.is_upper()', 'False'), ('+2:word.is_title()', 'False'),
                         ('+2:word.is_digit()', 'False'), ('+2:word.lower()', 'либерализации'),
                         ('-2:word.is_upper()', 'False'), ('-2:word.is_title()', 'False'),
                         ('-2:word.is_digit()', 'False'), ('-2:word.lower()', 'False'), ('-1:word.is_upper()', 'False'),
                         ('-1:word.is_title()', 'False'), ('-1:word.is_digit()', 'False'), ('-1:word.lower()', 'False'),
                         ('BOS', 'True'), ('EOS', 'False')]),
            OrderedDict([('bias', 1.0), ('word[-2]', 'ый'), ('word[-3]', 'ный'), ('word.is_upper()', 'False'),
                         ('word.is_title()', 'False'), ('word.is_digit()', 'False'), ('word.lower()', 'начальный'),
                         ('-1:word.is_upper()', 'True'), ('-1:word.is_title()', 'True'),
                         ('-1:word.is_digit()', 'False'), ('-1:word.lower()', 'в'), ('+1:word.is_upper()', 'False'),
                         ('+1:word.is_title()', 'False'), ('+1:word.is_digit()', 'False'),
                         ('+1:word.lower()', 'либерализации'), ('+2:word.is_upper()', 'False'),
                         ('+2:word.is_title()', 'False'), ('+2:word.is_digit()', 'False'), ('+2:word.lower()', '.'),
                         ('-2:word.is_upper()', 'False'), ('-2:word.is_title()', 'False'),
                         ('-2:word.is_digit()', 'False'), ('-2:word.lower()', 'False'), ('BOS', 'False'),
                         ('EOS', 'False')]),
            OrderedDict([('bias', 1.0), ('word[-2]', 'ии'), ('word[-3]', 'ции'), ('word.is_upper()', 'False'),
                         ('word.is_title()', 'False'), ('word.is_digit()', 'False'), ('word.lower()', 'либерализации'),
                         ('-2:word.is_upper()', 'True'), ('-2:word.is_title()', 'True'),
                         ('-2:word.is_digit()', 'False'), ('-2:word.lower()', 'в'), ('-1:word.is_upper()', 'False'),
                         ('-1:word.is_title()', 'False'), ('-1:word.is_digit()', 'False'),
                         ('-1:word.lower()', 'начальный'), ('+1:word.is_upper()', 'False'),
                         ('+1:word.is_title()', 'False'), ('+1:word.is_digit()', 'False'), ('+1:word.lower()', '.'),
                         ('+2:word.is_upper()', 'False'), ('+2:word.is_title()', 'False'),
                         ('+2:word.is_digit()', 'False'), ('+2:word.lower()', 'False'), ('BOS', 'False'),
                         ('EOS', 'False')]),
            OrderedDict([('bias', 1.0), ('word[-2]', 'False'), ('word[-3]', 'False'), ('word.is_upper()', 'False'),
                         ('word.is_title()', 'False'), ('word.is_digit()', 'False'), ('word.lower()', '.'),
                         ('-2:word.is_upper()', 'False'), ('-2:word.is_title()', 'False'),
                         ('-2:word.is_digit()', 'False'), ('-2:word.lower()', 'начальный'),
                         ('-1:word.is_upper()', 'False'), ('-1:word.is_title()', 'False'),
                         ('-1:word.is_digit()', 'False'), ('-1:word.lower()', 'либерализации'),
                         ('+1:word.is_upper()', 'False'), ('+1:word.is_title()', 'False'),
                         ('+1:word.is_digit()', 'False'), ('+1:word.lower()', 'False'), ('+2:word.is_upper()', 'False'),
                         ('+2:word.is_title()', 'False'), ('+2:word.is_digit()', 'False'), ('+2:word.lower()', 'False'),
                         ('BOS', 'False'), ('EOS', 'True')])
        ]
        true_result = pandas.DataFrame(true_result)
        fact_result = self.features.sentence2features(self.test_data)
        assert_frame_equal(fact_result.reset_index(drop=True), true_result.reset_index(drop=True))

    def test_sentence2labels(self):
        true_result = {
            'label': ['ADP', 'ADJ_Inan_Acc_Pos_Masc_Sing', 'NOUN_Inan_Gen_Fem_Sing', 'PUNCT']
        }
        true_result = pandas.DataFrame(true_result)
        fact_result = self.features.sentence2labels(self.test_data)
        assert_frame_equal(fact_result.reset_index(drop=True), true_result.reset_index(drop=True))

    def test_sentence2features_with_prefix(self):
        true_result = [
            [('+1:word.is_digit()', 'False'), ('+1:word.is_title()', 'False'), ('+1:word.is_upper()', 'False'),
             ('+1:word.lower()', 'начальный'), ('+1:word[-2]', 'ый'), ('+1:word[-3]', 'ный'), ('+1:word[2]', 'на'),
             ('+1:word[3]', 'нач'), ('+2:word.is_digit()', 'False'), ('+2:word.is_title()', 'False'),
             ('+2:word.is_upper()', 'False'), ('+2:word.lower()', 'либерализации'), ('+2:word[-2]', 'ии'),
             ('+2:word[-3]', 'ции'), ('+2:word[2]', 'ли'), ('+2:word[3]', 'либ'), ('-1:word.is_digit()', 'False'),
             ('-1:word.is_title()', 'False'), ('-1:word.is_upper()', 'False'), ('-1:word.lower()', 'False'),
             ('-1:word[-2]', 'False'), ('-1:word[-3]', 'False'), ('-1:word[2]', 'False'), ('-1:word[3]', 'False'),
             ('-2:word.is_digit()', 'False'), ('-2:word.is_title()', 'False'), ('-2:word.is_upper()', 'False'),
             ('-2:word.lower()', 'False'), ('-2:word[-2]', 'False'), ('-2:word[-3]', 'False'), ('-2:word[2]', 'False'),
             ('-2:word[3]', 'False'), ('BOS', 'True'), ('EOS', 'False'), ('bias', 1.0), ('word.is_digit()', 'False'),
             ('word.is_title()', 'True'), ('word.is_upper()', 'True'), ('word.lower()', 'в'), ('word[-2]', 'False'),
             ('word[-3]', 'False'), ('word[2]', 'False'), ('word[3]', 'False')],
            [('+1:word.is_digit()', 'False'), ('+1:word.is_title()', 'False'), ('+1:word.is_upper()', 'False'),
             ('+1:word.lower()', 'либерализации'), ('+1:word[-2]', 'ии'), ('+1:word[-3]', 'ции'), ('+1:word[2]', 'ли'),
             ('+1:word[3]', 'либ'), ('+2:word.is_digit()', 'False'), ('+2:word.is_title()', 'False'),
             ('+2:word.is_upper()', 'False'), ('+2:word.lower()', '.'), ('+2:word[-2]', 'False'),
             ('+2:word[-3]', 'False'), ('+2:word[2]', 'False'), ('+2:word[3]', 'False'), ('-1:word.is_digit()', 'False'),
             ('-1:word.is_title()', 'True'), ('-1:word.is_upper()', 'True'), ('-1:word.lower()', 'в'),
             ('-1:word[-2]', 'False'), ('-1:word[-3]', 'False'), ('-1:word[2]', 'False'), ('-1:word[3]', 'False'),
             ('-2:word.is_digit()', 'False'), ('-2:word.is_title()', 'False'), ('-2:word.is_upper()', 'False'),
             ('-2:word.lower()', 'False'), ('-2:word[-2]', 'False'), ('-2:word[-3]', 'False'), ('-2:word[2]', 'False'),
             ('-2:word[3]', 'False'), ('BOS', 'False'), ('EOS', 'False'), ('bias', 1.0), ('word.is_digit()', 'False'),
             ('word.is_title()', 'False'), ('word.is_upper()', 'False'), ('word.lower()', 'начальный'),
             ('word[-2]', 'ый'), ('word[-3]', 'ный'), ('word[2]', 'на'), ('word[3]', 'нач')],
            [('+1:word.is_digit()', 'False'), ('+1:word.is_title()', 'False'), ('+1:word.is_upper()', 'False'),
             ('+1:word.lower()', '.'), ('+1:word[-2]', 'False'), ('+1:word[-3]', 'False'), ('+1:word[2]', 'False'),
             ('+1:word[3]', 'False'), ('+2:word.is_digit()', 'False'), ('+2:word.is_title()', 'False'),
             ('+2:word.is_upper()', 'False'), ('+2:word.lower()', 'False'), ('+2:word[-2]', 'False'),
             ('+2:word[-3]', 'False'), ('+2:word[2]', 'False'), ('+2:word[3]', 'False'), ('-1:word.is_digit()', 'False'),
             ('-1:word.is_title()', 'False'), ('-1:word.is_upper()', 'False'), ('-1:word.lower()', 'начальный'),
             ('-1:word[-2]', 'ый'), ('-1:word[-3]', 'ный'), ('-1:word[2]', 'на'), ('-1:word[3]', 'нач'),
             ('-2:word.is_digit()', 'False'), ('-2:word.is_title()', 'True'), ('-2:word.is_upper()', 'True'),
             ('-2:word.lower()', 'в'), ('-2:word[-2]', 'False'), ('-2:word[-3]', 'False'), ('-2:word[2]', 'False'),
             ('-2:word[3]', 'False'), ('BOS', 'False'), ('EOS', 'False'), ('bias', 1.0), ('word.is_digit()', 'False'),
             ('word.is_title()', 'False'), ('word.is_upper()', 'False'), ('word.lower()', 'либерализации'),
             ('word[-2]', 'ии'), ('word[-3]', 'ции'), ('word[2]', 'ли'), ('word[3]', 'либ')],
            [('+1:word.is_digit()', 'False'), ('+1:word.is_title()', 'False'), ('+1:word.is_upper()', 'False'),
             ('+1:word.lower()', 'False'), ('+1:word[-2]', 'False'), ('+1:word[-3]', 'False'), ('+1:word[2]', 'False'),
             ('+1:word[3]', 'False'), ('+2:word.is_digit()', 'False'), ('+2:word.is_title()', 'False'),
             ('+2:word.is_upper()', 'False'), ('+2:word.lower()', 'False'), ('+2:word[-2]', 'False'),
             ('+2:word[-3]', 'False'), ('+2:word[2]', 'False'), ('+2:word[3]', 'False'), ('-1:word.is_digit()', 'False'),
             ('-1:word.is_title()', 'False'), ('-1:word.is_upper()', 'False'), ('-1:word.lower()', 'либерализации'),
             ('-1:word[-2]', 'ии'), ('-1:word[-3]', 'ции'), ('-1:word[2]', 'ли'), ('-1:word[3]', 'либ'),
             ('-2:word.is_digit()', 'False'), ('-2:word.is_title()', 'False'), ('-2:word.is_upper()', 'False'),
             ('-2:word.lower()', 'начальный'), ('-2:word[-2]', 'ый'), ('-2:word[-3]', 'ный'), ('-2:word[2]', 'на'),
             ('-2:word[3]', 'нач'), ('BOS', 'False'), ('EOS', 'True'), ('bias', 1.0), ('word.is_digit()', 'False'),
             ('word.is_title()', 'False'), ('word.is_upper()', 'False'), ('word.lower()', '.'), ('word[-2]', 'False'),
             ('word[-3]', 'False'), ('word[2]', 'False'), ('word[3]', 'False')]]
        fact_result = self.features_with_prefix.sentence2features(self.test_data)
        self.assertEqual([sorted(list(el.items())) for el in fact_result[0]], true_result)

if __name__ == '__main__':
    unittest.main(verbosity=2)