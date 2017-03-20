# -*- coding: utf-8 -*-
#
# Скрипт ориентирован на использование в Python 3.*
#
# Скрипт содержит в себе класс MorphoParser для парсинга различных типов корпусов.

import os
import pickle
import pprint


class MorphoParser:
    """
    Класс для подготовки данных.
    """

    def extract_grammemes(self, grammem_string):
        """
        Парсинг FEATS.
        :return:
        """

        return '_'.join([el.split('=')[-1] for el in grammem_string.split('|')]).replace('__', '')

    def form_token_label_pos(self, data, labels_format):
        """
        Формирование объекта и класса для POS-классификатора.
        :param data:
        :param labels_format:
        :return:
        """

        splitting_data = [data[index] for index in labels_format]
        word, label = splitting_data[0], self.extract_grammemes('|'.join(splitting_data[1:]))
        # для классов morpho_subtags: NUM _; ADP _;
        if label == '_':
            label = label.replace(label, 'O')
        return word, label

    def form_token_label_binary(self, data, labels_format):
        """
        Формирование объекта и класса для бинарного классификатора.
        :param data:
        :param labels_format:
        :return:
        """

        word, label = data[1], [el.split('=')[1] if data[5] != '_' else 'O' for el in data[5].split('|')]
        if labels_format in label:
            label = labels_format
        else:
            label = 'O'
        return word, label

    def form_token_label_grammatical_categ(self, data, labels_format, label_):
        """
        Формирование объекта и класса для классификаторов грамматических категорий.
        :param data:
        :param label_options:
        :return:
        """

        word = data[1]
        label = None
        if data[labels_format[0]] == '_':
            label = 'O'
        else:
            gramm_cat = [el.split('=') for el in data[labels_format[1]].split('|')]
            label = [el[1] for el in gramm_cat if el[0] == label_]
            if label:
                label = label[0]
            else:
                label = 'O'
        return word, label

    def parse(self, sentences, labels_format, label_):
        """
        Создать для каждого токена кортежи формата: (словоформа, грамматический класс).
        """

        stop_str_rnc = [
            ['==newfile=='],
            ['==> fiction.xhtml <=='],
            ['==> public.xhtml <=='],
            ['==> science.xhtml <=='],
            ['==> speech.xhtml <==']
        ]
        all_sentences_with_data = list()
        for sentence in sentences:
            if not sentence.strip():
                continue
            all_words_in_sentence = list()
            for data in sentence.rstrip().split('\n'):
                data = data.split('\t')
                if data not in stop_str_rnc:
                    if label_ == 'POS':
                        word, pos = self.form_token_label_pos(data, labels_format)
                        all_words_in_sentence.append((word, pos))
                    elif label_ == "POS&GRAMMAR":
                        word, pos_grammar = self.form_token_label_pos(data, labels_format)
                        all_words_in_sentence.append((word, pos_grammar))
                    else:
                        word, grammar_categ = self.form_token_label_grammatical_categ(data, labels_format, label_)
                        all_words_in_sentence.append((word, grammar_categ))
            all_sentences_with_data.append(all_words_in_sentence)
        return all_sentences_with_data
