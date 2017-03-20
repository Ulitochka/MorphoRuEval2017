# -*- coding: utf-8 -*-
#
# Скрипт ориентирован на использование в Python 3.*
#
# Скрипт содержит в себе класс StatWorker.

import os

from tabulate import tabulate
from collections import Counter
import csv


class StatWorker:
    def __init__(self, features=None):
        self.lower_upper_diff = features.get("LOWER_UPPER_DIFF")

        self.count_unique_labels = set()
        self.sent_length = list()
        self.count_unique_tokens = set()
        self.count_tokens_ = 0
        self.average_sent_length = 0
        self.max_sent_length = 0
        self.min_sent_length = 0

        # for lstm;
        self.unique_symbols = set()
        self.max_token_length = list()

    def count_elements(self, data=None):
        """
        Подсчет количества различных объектов.
        :param data:
        :return:
        """

        for sent in data:
            self.count_tokens_ += len(sent)
            self.sent_length.append(len(sent))
            for token in sent:
                self.count_unique_tokens.add(token[0].lower())
                self.count_unique_labels.add(token[-1])

                self.max_token_length.append(len(token[0]))
                if self.lower_upper_diff:
                    self.unique_symbols |= set(token[0])
                else:
                    self.unique_symbols |= set(token[0].lower())
        del data

    def create_report(self, visual=True):
        """
        Создание отчета.
        :return:
        """

        self.average_sent_length = int(sum(self.sent_length) / len(self.sent_length))
        count_sent = len(self.sent_length)
        self.max_sent_length = max(self.sent_length)
        self.min_sent_length = min(self.sent_length)
        self.count_unique_tokens_ = len(self.count_unique_tokens)
        self.count_unique_labels_ = len(self.count_unique_labels)

        if visual:
            print(tabulate([
                [
                    count_sent,
                    self.average_sent_length,
                    self.max_sent_length,
                    self.min_sent_length,
                    self.count_tokens_,
                    self.count_unique_tokens_,
                    self.count_unique_labels_
                ]],
               headers=[
                   'count_sent',
                   'average_sent_length',
                   'max_sent_length',
                   'min_sent_length',
                   'count_tokens',
                   'count_unique_tokens',
                   'count_unique_labels']))
