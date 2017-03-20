import unittest
import pprint

from tools.parsers.Parser import MorphoParser
import numpy
from scipy import sparse


class TestMorphoParser(unittest.TestCase):
    def setUp(self):
        self.parser = MorphoParser()

        self.test_data_UD = ['1\tАнкета\tанкета\tNOUN\t_\tAnimacy=Inan|Case=Nom|Gender=Fem|Number=Sing\t0\troot\t_\t_\n'
                          '2\t.\t.\tPUNCT\t.\t_\t1\tpunct\t_\t_',
                          '1\tНачальник\tначальник\tNOUN\t_\tAnimacy=Anim|Case=Nom|Gender=Masc|Number=Sing\t8\tnsubj\t_\t_\n'
                          '2\tобластного\tобластной\tADJ\t_\tCase=Gen|Degree=Pos|Gender=Neut|Number=Sing\t3\tamod\t_\t_\n'
                          '3\tуправления\tуправление\tNOUN\t_\tAnimacy=Inan|Case=Gen|Gender=Neut|Number=Sing\t1\tnmod\t_\t_\n'
                          '4\tсвязи\tсвязь\tNOUN\t_\tAnimacy=Inan|Case=Gen|Gender=Fem|Number=Sing\t3\tnmod\t_\t_\n'
                          '5\tСемен\tсемен\tPROPN\t_\tAnimacy=Anim|Case=Nom|Gender=Masc|Number=Sing\t1\tappos\t_\t_\n'
                          '6\tЕремеевич\tеремеевич\tPROPN\t_\tAnimacy=Anim|Case=Nom|Gender=Masc|Number=Sing\t5\tname\t_\t_\n'
                          '7\tбыл\tбыть\tAUX\t_\tAspect=Imp|Gender=Masc|Mood=Ind|Number=Sing|Tense=Past|VerbForm=Fin|Voice=Act\t8\tcop\t_\t_\n'
                          '8\tчеловек\tчеловек\tNOUN\t_\tAnimacy=Anim|Case=Nom|Gender=Masc|Number=Sing\t0\troot\t_\t_\n'
                          '9\tпростой\tпростой\tADJ\t_\tCase=Nom|Degree=Pos|Gender=Masc|Number=Sing\t8\tamod\t_\t_\n'
                          '10\t,\t,\tPUNCT\t,\t_\t9\tpunct\t_\t_\n'
                          '11\tприходил\tприходить\tVERB\t_\tAspect=Imp|Gender=Masc|Mood=Ind|Number=Sing|Tense=Past|VerbForm=Fin|Voice=Act\t8\tconj\t_\t_\n'
                          '12\tна\tна\tADP\t_\t_\t13\tcase\t_\t_\n13\tработу\tработа\tNOUN\t_\tAnimacy=Inan|Case=Acc|Gender=Fem|Number=Sing\t11\tnmod\t_\t_\n'
                          '14\tвсегда\tвсегда\tADV\t_\tDegree=Pos\t11\tadvmod\t_\t_\n'
                          '15\tвовремя\tвовремя\tADV\t_\tDegree=Pos\t11\tadvmod\t_\t_\n'
                          '16\t,\t,\tPUNCT\t,\t_\t15\tpunct\t_\t_\n'
                          '17\tздоровался\tздороваться\tVERB\t_\tAspect=Imp|Gender=Masc|Mood=Ind|Number=Sing|Tense=Past|VerbForm=Fin|Voice=Act\t8\tconj\t_\t_\n'
                          '18\tс\tс\tADP\t_\t_\t19\tcase\t_\t_\n'
                          '19\tсекретаршей\tсекретарша\tNOUN\t_\tAnimacy=Anim|Case=Ins|Gender=Fem|Number=Sing\t17\tnmod\t_\t_\n'
                          '20\tза\tза\tADP\t_\t_\t21\tcase\t_\t_\n'
                          '21\tруку\tрука\tNOUN\t_\tAnimacy=Inan|Case=Acc|Gender=Fem|Number=Sing\t17\tnmod\t_\t_\n'
                          '22\tи\tи\tCONJ\t_\t_\t8\tcc\t_\t_\n'
                          '23\tиногда\tиногда\tADV\t_\tDegree=Pos\t25\tadvmod\t_\t_\n'
                          '24\tдаже\tдаже\tPART\t_\t_\t25\tadvmod\t_\t_\n'
                          '25\tписал\tписать\tVERB\t_\tAspect=Imp|Gender=Masc|Mood=Ind|Number=Sing|Tense=Past|VerbForm=Fin|Voice=Act\t8\tconj\t_\t_\n'
                          '26\tв\tв\tADP\t_\t_\t27\tcase\t_\t_\n'
                          '27\tстенгазету\tстенгазета\tNOUN\t_\tAnimacy=Inan|Case=Acc|Gender=Fem|Number=Sing\t25\tnmod\t_\t_\n'
                          '28\tзаметки\tзаметка\tNOUN\t_\tAnimacy=Inan|Case=Acc|Gender=Fem|Number=Plur\t25\tdobj\t_\t_\n'
                          '29\tпод\tпод\tADP\t_\t_\t30\tcase\t_\t_\n'
                          '30\tпсевдонимом\tпсевдоним\tNOUN\t_\tAnimacy=Inan|Case=Ins|Gender=Masc|Number=Sing\t25\tnmod\t_\t_\n'
                          '31\t"\t"\tPUNCT\t"\t_\t32\tpunct\t_\t_\n'
                          '32\tМуха\tмуха\tPROPN\t_\tAnimacy=Anim|Case=Nom|Gender=Fem|Number=Sing\t30\tnmod\t_\t_\n'
                          '33\t"\t"\tPUNCT\t"\t_\t32\tpunct\t_\t_\n'
                          '34\t.\t.\tPUNCT\t.\t_\t8\tpunct\t_\t_']

        self.test_data_GCRY = ['1\tУкрепленные\tукреплённый\tADJ\tCase=Nom|Degree=Pos|Number=Plur\n'
                               '2\tна\tна\tADP\t_\n'
                               '3\tкрючьях\tкрюк\tNOUN\tAnimacy=Inan|Case=Loc|Gender=Masc|Number=Plur\n'
                               '4\tфакелы\tфакел\tNOUN\tAnimacy=Inan|Case=Nom|Gender=Masc|Number=Plur\n'
                               '5\tосвещали\tосвещать\tVERB\tMood=Ind|Number=Plur|Tense=Past|VerbForm=Fin|Voice=Act\n'
                               '6\tдревние\tдревний\tADJ\tCase=Acc|Degree=Pos|Number=Plur\n'
                               '7\tстены\tстена\tNOUN\tAnimacy=Inan|Case=Acc|Gender=Fem|Number=Plur\n'
                               '8\tиз\tиз\tADP\t_\n'
                               '9\tдикого\tдикий\tADJ\tCase=Gen|Degree=Pos|Gender=Masc|Number=Sing\n'
                               '10\tкамня\tкамень\tNOUN\tAnimacy=Inan|Case=Gen|Gender=Masc|Number=Sing\n'
                               '11\tи\tи\tCONJ\t_\n'
                               '12\tтяжелые\tтяжёлый\tADJ\tCase=Acc|Degree=Pos|Number=Plur\n'
                               '13\tплиты\tплита\tNOUN\tAnimacy=Inan|Case=Acc|Gender=Fem|Number=Plur\n'
                               '14\tровного\tровный\tADJ\tCase=Gen|Degree=Pos|Gender=Masc|Number=Sing\n'
                               '15\tпола\tпол\tNOUN\tAnimacy=Inan|Case=Gen|Gender=Masc|Number=Sing\n'
                               '16\t.\t.\tPUNCT\t_',
                               '1\tТе\tтот\tDET\tCase=Nom|Number=Plur\n'
                               '2\t,\t,\tPUNCT\t_\n'
                               '3\tкто\tкто\tPRON\tCase=Nom|Number=Sing\n'
                               '4\tне\tне\tPART\t_\n'
                               '5\tпонимает\tпонимать\tVERB\tMood=Ind|Number=Sing|Person=3|Tense=Notpast|VerbForm=Fin|Voice=Act\n'
                               '6\tтого\tто\tPRON\tCase=Gen|Gender=Neut|Number=Sing\n'
                               '7\t,\t,\tPUNCT\t_\n'
                               '8\tчто\tчто\tPRON\tCase=Nom|Gender=Neut|Number=Sing\n'
                               '9\tс\tс\tADP\t_\n'
                               '10\tними\tон\tPRON\tCase=Ins|Number=Plur|Person=3\n'
                               '11\tпроисходит\tпроисходить\tVERB\tMood=Ind|Number=Sing|Person=3|Tense=Notpast|VerbForm=Fin|Voice=Act\n'
                               '12\t,\t,\tPUNCT\t_\n'
                               '13\tстановятся\tстановиться\tVERB\tMood=Ind|Number=Plur|Person=3|Tense=Notpast|VerbForm=Fin|Voice=Mid\n'
                               '14\tневротиками\tневротик\tNOUN\tAnimacy=Anim|Case=Ins|Gender=Masc|Number=Plur\n'
                               '15\t,\t,\tPUNCT\t_\n'
                               '16\tа\tа\tCONJ\t_\n'
                               '17\tзатем\tзатем\tADV\tDegree=Pos\n'
                               '18\t,\t,\tPUNCT\t_\n'
                               '19\tпри\tпри\tADP\t_\n'
                               '20\tпрогрессировании\tпрогрессирование\tNOUN\tAnimacy=Inan|Case=Loc|Gender=Neut|Number=Sing\n'
                               '21\tдепрессии\tдепрессия\tNOUN\tAnimacy=Inan|Case=Gen|Gender=Fem|Number=Sing\n'
                               '22\t,\t,\tPUNCT\t_\n'
                               '23\tв\tв\tADP\t_\n'
                               '24\tтой\tтот\tDET\tCase=Loc|Gender=Fem|Number=Sing\n'
                               '25\tили\tили\tCONJ\t_\n'
                               '26\tиной\tиной\tDET\tCase=Loc|Gender=Fem|Number=Sing\n'
                               '27\tстепени\tстепень\tNOUN\tAnimacy=Inan|Case=Loc|Gender=Fem|Number=Sing\n'
                               '28\t,\t,\tPUNCT\t_\n'
                               '29\tи\tи\tPART\t_\n'
                               '30\tпсихически\tпсихически\tADV\tDegree=Pos\n'
                               '31\tбольными\tбольной\tADJ\tCase=Ins|Degree=Pos|Number=Plur\n'
                               '32\tлюдьми\tчеловек\tNOUN\tAnimacy=Anim|Case=Ins|Gender=Masc|Number=Plur\n'
                               '33\t.\t.\tPUNCT\t_']

        self.test_data_OPENCORP = ['1\tМузеи\tМУЗЕЙ\tNOUN\t_\tAnimacy=Inan|Case=Nom|Gender=Masc|Number=Plur\t_\t_\t_\t_',
                                   '1\tСамых\tСАМЫЙ\tDET\t_\tAnimacy=Anim|Case=Acc|Number=Plur\t_\t_\t_\t_\n'
                                   '2\tспособных\tСПОСОБНЫЙ\tADJ\t_\tAnimacy=Anim|Case=Acc|Number=Plur\t_\t_\t_\t_\n'
                                   '3\tон\tОН\tPRON\t_\tCase=Nom|Gender=Masc|Number=Sing|Person=3\t_\t_\t_\t_\n'
                                   '4\tпривлёк\tПРИВЛЕЧЬ\tVERB\t_\tAspect=Perf|Gender=Masc|Mood=Ind|Number=Sing|Tense=Past|VerbForm=Fin\t_\t_\t_\t_\n'
                                   '5\tна\tНА\tADP\t_\t_\t_\t_\t_\t_\n'
                                   '6\tслужбу\tСЛУЖБА\tNOUN\t_\tAnimacy=Inan|Case=Acc|Gender=Fem|Number=Sing\t_\t_\t_\t_\n'
                                   '7\t,\t,\tPUNCT\t_\t_\t_\t_\t_\t_\n'
                                   '8\tрасплачиваясь\tРАСПЛАЧИВАТЬСЯ\tVERB\t_\tAspect=Imp|Tense=Notpast|VerbForm=Conv|Voice=Mid\t_\t_\t_\t_\n'
                                   '9\tземлёй\tЗЕМЛЯ\tNOUN\t_\tAnimacy=Inan|Case=Ins|Gender=Fem|Number=Sing\t_\t_\t_\t_\n'
                                   '10\t,\t,\tPUNCT\t_\t_\t_\t_\t_\t_\n'
                                   '11\tа\tА\tCONJ\t_\t_\t_\t_\t_\t_\n'
                                   '12\tостальных\tОСТАЛЬНОЙ\tDET\t_\tAnimacy=Anim|Case=Acc|Number=Plur\t_\t_\t_\t_\n'
                                   '13\tсплавил\tСПЛАВИТЬ\tVERB\t_\tAspect=Perf|Gender=Masc|Mood=Ind|Number=Sing|Tense=Past|VerbForm=Fin\t_\t_\t_\t_\n'
                                   '14\tв\tВ\tADP\t_\t_\t_\t_\t_\t_\n'
                                   '15\tКонстантинополь\tКОНСТАНТИНОПОЛЬ\tPROPN\t_\tAnimacy=Inan|Case=Acc|Gender=Masc|Number=Sing\t_\t_\t_\t_\n'
                                   '16\t.\t.\tPUNCT\t_\t_\t_\t_\t_\t_']

        self.test_data_SYNTAG = ['1\tА\tа\tCONJ\t_\n'
                                 '2\tтеперь\tтеперь\tADV\tDegree=Pos\n'
                                 '3\tони\tон\tPRON\tCase=Nom|Number=Plur|Person=3\n'
                                 '4\tвозят\tвозить\tVERB\tMood=Ind|Number=Plur|Person=3|Tense=Notpast|VerbForm=Fin|Voice=Act\n'
                                 '5\tна\tна\tADP\t_\n'
                                 '6\tоколоземные\tоколоземный\tADJ\tCase=Acc|Degree=Pos|Number=Plur\n'
                                 '7\tорбиты\tорбита\tNOUN\tAnimacy=Inan|Case=Acc|Gender=Fem|Number=Plur\n'
                                 '8\tтуристов\tтурист\tNOUN\tAnimacy=Anim|Case=Acc|Gender=Masc|Number=Plur\n'
                                 '9\t-\t-\tPUNCT\t_\n'
                                 '10\tбогатых\tбогатый\tADJ\tCase=Acc|Degree=Pos|Number=Plur\n'
                                 '11\tиностранцев\tиностранец\tNOUN\tAnimacy=Anim|Case=Acc|Gender=Masc|Number=Plur\n'
                                 '12\t.\t.\tPUNCT\t_',
                                 '1\t"\t"\tPUNCT\t_\n'
                                 '2\tЭта\tэтот\tDET\tCase=Nom|Gender=Fem|Number=Sing\n'
                                 '3\tтема\tтема\tNOUN\tAnimacy=Inan|Case=Nom|Gender=Fem|Number=Sing\n'
                                 '4\tкасается\tкасаться\tVERB\tMood=Ind|Number=Sing|Person=3|Tense=Notpast|VerbForm=Fin|Voice=Mid\n'
                                 '5\tвсех\tвсе\tPRON\tCase=Gen|Number=Plur\n'
                                 '6\tи\tи\tCONJ\t_\n'
                                 '7\tкаждого\tкаждый\tADJ\tCase=Gen|Degree=Pos|Gender=Masc|Number=Sing\n'
                                 '8\t,\t,\tPUNCT\t_\n'
                                 '9\tи\tи\tCONJ\t_\n'
                                 '10\tпоэтому\tпоэтому\tADV\tDegree=Pos\n'
                                 '11\tинтересна\tинтересный\tADJ\tDegree=Pos|Gender=Fem|Number=Sing|Variant=Short\n'
                                 '12\tвсегда\tвсегда\tADV\tDegree=Pos\n'
                                 '13\t"\t"\tPUNCT\t_\n'
                                 '14\t,\t,\tPUNCT\t_\n'
                                 '15\t-\t-\tPUNCT\t_\n'
                                 '16\tсказал\tсказать\tVERB\tGender=Masc|Mood=Ind|Number=Sing|Tense=Past|VerbForm=Fin|Voice=Act\n'
                                 '17\tСамгин\tсамгин\tNOUN\tAnimacy=Anim|Case=Nom|Gender=Masc|Number=Sing\n'
                                 '18\t.\t.\tPUNCT\t_']

        self.test_data_RNC = ['\tГлавный\tглавный\tADJ\tCase=Nom|Gender=Masc|Number=Sing|Variant=Full\t_\n'
                              '\tучёный\tученый\tADJ\tCase=Nom|Gender=Masc|Number=Sing|Variant=Full\t_\n'
                              '\tсекретарь\tсекретарь\tNOUN\tAnimacy=Anim|Case=Nom|Gender=Masc|Number=Sing\t_\n'
                              '\tОтделения\tотделение\tNOUN\tAnimacy=Inan|Case=Gen|Gender=Neut|Number=Sing\t_\n'
                              '\tчлен-корреспондент\tчлен-корреспондент\tNOUN\tAnimacy=Anim|Case=Nom|Gender=Masc|Number=Sing\t_\n'
                              '\tРАН\tРАН\tNOUN\tAnimacy=Inan|Case=Gen|Gender=Fem|Number=Sing\tAbbr=Yes|Decl=Zero\n'
                              '\tВ\tВ\tNOUN\t_\tAbbr=Init|Abbr=Yes\n'
                              '\t.\t.\tPUNCT\t_\t_\n'
                              '\tМ\tМ\tNOUN\t_\tAbbr=Init|Abbr=Yes\n'
                              '\t.\t.\tPUNCT\t_\t_\n'
                              '\tФомин\tФомин\tNOUN\tAnimacy=Anim|Case=Nom|Gender=Masc|Number=Sing\tNameType=Sur',
                              '\tВ\tв\tADP\t_\t_\n\tдетстве\tдетство\tNOUN\tAnimacy=Inan|Case=Loc|Gender=Neut|Number=Sing\t_\n'
                              '\tсилой\tсила\tNOUN\tAnimacy=Inan|Case=Ins|Gender=Fem|Number=Sing\t_\n'
                              '\tзаставляли\tзаставлять\tVERB\tMood=Ind|Number=Plur|Tense=Past|VerbForm=Fin|Voice=Act\tSubcat=Tran|Aspect=Imp\n'
                              '\tходить\tходить\tVERB\tVerbForm=Inf|Voice=Act\tSubcat=Intr|Aspect=Imp\n'
                              '\tв\tв\tADP\t_\t_\n'
                              '\tанглийскую\tанглийский\tADJ\tCase=Acc|Gender=Fem|Number=Sing|Variant=Full\t_\n'
                              '\tгруппу\tгруппа\tNOUN\tAnimacy=Inan|Case=Acc|Gender=Fem|Number=Sing\t_\n'
                              '\t,\t,\tPUNCT\t_\t_\n'
                              '\tа\tа\tCONJ\t_\t_\n'
                              '\tпотом\tпотом\tADV\tDegree=Pos\t_\n'
                              '\tпристрастился\tпристраститься\tVERB\tGender=Masc|Mood=Ind|Number=Sing|Tense=Past|VerbForm=Fin|Voice=Mid\tAspect=Perf\n'
                              '\tк\tк\tADP\t_\t_\n'
                              '\tdetective\tdetective\tX\t_\t_\n'
                              '\tstory\tstory\tX\t_\t_\n'
                              '\t.\t.\tPUNCT\t_\t_']

    def test_parser_RNC_pos(self):
        true_result = [
            [('Главный', 'ADJ'), ('учёный', 'ADJ'), ('секретарь', 'NOUN'), ('Отделения', 'NOUN'),
             ('член-корреспондент', 'NOUN'), ('РАН', 'NOUN'), ('В', 'NOUN'), ('.', 'PUNCT'), ('М', 'NOUN'),
             ('.', 'PUNCT'), ('Фомин', 'NOUN')],
            [('В', 'ADP'), ('детстве', 'NOUN'), ('силой', 'NOUN'), ('заставляли', 'VERB'), ('ходить', 'VERB'),
             ('в', 'ADP'), ('английскую', 'ADJ'), ('группу', 'NOUN'), (',', 'PUNCT'), ('а', 'CONJ'), ('потом', 'ADV'),
             ('пристрастился', 'VERB'), ('к', 'ADP'), ('detective', 'X'), ('story', 'X'), ('.', 'PUNCT')]]
        fact_result = self.parser.parse(self.test_data_RNC, [1, 3], "POS")
        self.assertEqual(fact_result, true_result)

    def test_parser_RNC_all(self):
        true_result = [
            [('Главный', 'ADJ_Nom_Masc_Sing_Full'), ('учёный', 'ADJ_Nom_Masc_Sing_Full'),
             ('секретарь', 'NOUN_Anim_Nom_Masc_Sing'), ('Отделения', 'NOUN_Inan_Gen_Neut_Sing'),
             ('член-корреспондент', 'NOUN_Anim_Nom_Masc_Sing'), ('РАН', 'NOUN_Inan_Gen_Fem_Sing'),
             ('В', 'NOUN'), ('.', 'PUNCT'), ('М', 'NOUN'), ('.', 'PUNCT'), ('Фомин', 'NOUN_Anim_Nom_Masc_Sing')],
            [('В', 'ADP'), ('детстве', 'NOUN_Inan_Loc_Neut_Sing'), ('силой', 'NOUN_Inan_Ins_Fem_Sing'),
             ('заставляли', 'VERB_Ind_Plur_Past_Fin_Act'), ('ходить', 'VERB_Inf_Act'), ('в', 'ADP'),
             ('английскую', 'ADJ_Acc_Fem_Sing_Full'), ('группу', 'NOUN_Inan_Acc_Fem_Sing'), (',', 'PUNCT'),
             ('а', 'CONJ'), ('потом', 'ADV_Pos'), ('пристрастился', 'VERB_Masc_Ind_Sing_Past_Fin_Mid'), ('к', 'ADP'),
             ('detective', 'X'), ('story', 'X'), ('.', 'PUNCT')]]
        fact_result = self.parser.parse(self.test_data_RNC, [1, 3, 4], "POS&GRAMMAR")
        self.assertEqual(fact_result, true_result)

    def test_parser_RNC_number(self):
        true_result = [
            [('Главный', 'Sing'), ('учёный', 'Sing'), ('секретарь', 'Sing'), ('Отделения', 'Sing'),
             ('член-корреспондент', 'Sing'), ('РАН', 'Sing'), ('В', 'O'), ('.', 'O'), ('М', 'O'), ('.', 'O'),
             ('Фомин', 'Sing')],
            [('В', 'O'), ('детстве', 'Sing'), ('силой', 'Sing'), ('заставляли', 'Plur'), ('ходить', 'O'), ('в', 'O'),
             ('английскую', 'Sing'), ('группу', 'Sing'), (',', 'O'), ('а', 'O'), ('потом', 'O'), ('пристрастился', 'Sing'),
             ('к', 'O'), ('detective', 'O'), ('story', 'O'), ('.', 'O')]]
        fact_result = self.parser.parse(self.test_data_RNC, [1, 4], "Number")
        self.assertEqual(fact_result, true_result)

    def test_parser_SYNTAG_pos(self):
        true_result = [
            [('А', 'CONJ'), ('теперь', 'ADV'), ('они', 'PRON'), ('возят', 'VERB'), ('на', 'ADP'),
             ('околоземные', 'ADJ'), ('орбиты', 'NOUN'), ('туристов', 'NOUN'), ('-', 'PUNCT'), ('богатых', 'ADJ'),
             ('иностранцев', 'NOUN'), ('.', 'PUNCT')],
            [('"', 'PUNCT'), ('Эта', 'DET'), ('тема', 'NOUN'), ('касается', 'VERB'), ('всех', 'PRON'), ('и', 'CONJ'),
             ('каждого', 'ADJ'), (',', 'PUNCT'), ('и', 'CONJ'), ('поэтому', 'ADV'), ('интересна', 'ADJ'),
             ('всегда', 'ADV'), ('"', 'PUNCT'), (',', 'PUNCT'), ('-', 'PUNCT'), ('сказал', 'VERB'), ('Самгин', 'NOUN'),
             ('.', 'PUNCT')]
        ]
        fact_result = self.parser.parse(self.test_data_SYNTAG, [1, 3], "POS")
        self.assertEqual(fact_result, true_result)

    def test_parser_SYNTAG_all(self):

        true_result = [
            [('А', 'CONJ'), ('теперь', 'ADV_Pos'), ('они', 'PRON_Nom_Plur_3'),
             ('возят', 'VERB_Ind_Plur_3_Notpast_Fin_Act'), ('на', 'ADP'), ('околоземные', 'ADJ_Acc_Pos_Plur'),
             ('орбиты', 'NOUN_Inan_Acc_Fem_Plur'), ('туристов', 'NOUN_Anim_Acc_Masc_Plur'), ('-', 'PUNCT'),
             ('богатых', 'ADJ_Acc_Pos_Plur'), ('иностранцев', 'NOUN_Anim_Acc_Masc_Plur'), ('.', 'PUNCT')],
            [('"', 'PUNCT'), ('Эта', 'DET_Nom_Fem_Sing'), ('тема', 'NOUN_Inan_Nom_Fem_Sing'),
             ('касается', 'VERB_Ind_Sing_3_Notpast_Fin_Mid'), ('всех', 'PRON_Gen_Plur'), ('и', 'CONJ'),
             ('каждого', 'ADJ_Gen_Pos_Masc_Sing'), (',', 'PUNCT'), ('и', 'CONJ'), ('поэтому', 'ADV_Pos'),
             ('интересна', 'ADJ_Pos_Fem_Sing_Short'), ('всегда', 'ADV_Pos'), ('"', 'PUNCT'), (',', 'PUNCT'),
             ('-', 'PUNCT'), ('сказал', 'VERB_Masc_Ind_Sing_Past_Fin_Act'), ('Самгин', 'NOUN_Anim_Nom_Masc_Sing'),
             ('.', 'PUNCT')]
        ]
        fact_result = self.parser.parse(self.test_data_SYNTAG, [1, 3, 4], "POS&GRAMMAR")
        self.assertEqual(fact_result, true_result)

    def test_parser_SYNTAG_number(self):

        true_result = [
            [('А', 'O'), ('теперь', 'O'), ('они', 'Plur'), ('возят', 'Plur'), ('на', 'O'), ('околоземные', 'Plur'),
             ('орбиты', 'Plur'), ('туристов', 'Plur'), ('-', 'O'), ('богатых', 'Plur'), ('иностранцев', 'Plur'),
             ('.', 'O')],
            [('"', 'O'), ('Эта', 'Sing'), ('тема', 'Sing'), ('касается', 'Sing'), ('всех', 'Plur'), ('и', 'O'),
             ('каждого', 'Sing'), (',', 'O'), ('и', 'O'), ('поэтому', 'O'), ('интересна', 'Sing'), ('всегда', 'O'),
             ('"', 'O'), (',', 'O'), ('-', 'O'), ('сказал', 'Sing'), ('Самгин', 'Sing'), ('.', 'O')]
        ]

        fact_result = self.parser.parse(self.test_data_SYNTAG, [1, 4], "Number")
        self.assertEqual(fact_result, true_result)

    def test_parser_OPENCORP_pos(self):
        true_result = [
            [('Музеи', 'NOUN')],
            [('Самых', 'DET'), ('способных', 'ADJ'), ('он', 'PRON'), ('привлёк', 'VERB'), ('на', 'ADP'),
             ('службу', 'NOUN'), (',', 'PUNCT'), ('расплачиваясь', 'VERB'), ('землёй', 'NOUN'), (',', 'PUNCT'),
             ('а', 'CONJ'), ('остальных', 'DET'), ('сплавил', 'VERB'), ('в', 'ADP'), ('Константинополь', 'PROPN'),
             ('.', 'PUNCT')]
        ]
        fact_result = self.parser.parse(self.test_data_OPENCORP, [1, 3], "POS")
        self.assertEqual(fact_result, true_result)

    def test_parser_OPENCORP_all(self):

        true_result = [
            [('Музеи', 'NOUN_Inan_Nom_Masc_Plur')],
            [('Самых', 'DET_Anim_Acc_Plur'), ('способных', 'ADJ_Anim_Acc_Plur'), ('он', 'PRON_Nom_Masc_Sing_3'),
             ('привлёк', 'VERB_Perf_Masc_Ind_Sing_Past_Fin'), ('на', 'ADP'), ('службу', 'NOUN_Inan_Acc_Fem_Sing'),
             (',', 'PUNCT'), ('расплачиваясь', 'VERB_Imp_Notpast_Conv_Mid'), ('землёй', 'NOUN_Inan_Ins_Fem_Sing'),
             (',', 'PUNCT'), ('а', 'CONJ'), ('остальных', 'DET_Anim_Acc_Plur'),
             ('сплавил', 'VERB_Perf_Masc_Ind_Sing_Past_Fin'), ('в', 'ADP'),
             ('Константинополь', 'PROPN_Inan_Acc_Masc_Sing'), ('.', 'PUNCT')]
        ]
        fact_result = self.parser.parse(self.test_data_OPENCORP, [1, 3, 5], "POS&GRAMMAR")
        self.assertEqual(fact_result, true_result)

    def test_parser_OPENCORP_mood(self):

        true_result = [
            [('Музеи', 'O')],
            [('Самых', 'O'), ('способных', 'O'), ('он', 'O'), ('привлёк', 'Ind'), ('на', 'O'), ('службу', 'O'),
             (',', 'O'), ('расплачиваясь', 'O'), ('землёй', 'O'), (',', 'O'), ('а', 'O'), ('остальных', 'O'),
             ('сплавил', 'Ind'), ('в', 'O'), ('Константинополь', 'O'), ('.', 'O')]
        ]
        fact_result = self.parser.parse(self.test_data_OPENCORP, [1, 5], "Mood")
        self.assertEqual(fact_result, true_result)

    def test_parser_GCRY_pos(self):
        true_result = [
            [('Укрепленные', 'ADJ'), ('на', 'ADP'), ('крючьях', 'NOUN'), ('факелы', 'NOUN'), ('освещали', 'VERB'),
             ('древние', 'ADJ'), ('стены', 'NOUN'), ('из', 'ADP'), ('дикого', 'ADJ'), ('камня', 'NOUN'),
             ('и', 'CONJ'), ('тяжелые', 'ADJ'), ('плиты', 'NOUN'), ('ровного', 'ADJ'), ('пола', 'NOUN'),
             ('.', 'PUNCT')],
            [('Те', 'DET'), (',', 'PUNCT'), ('кто', 'PRON'), ('не', 'PART'), ('понимает', 'VERB'), ('того', 'PRON'),
             (',', 'PUNCT'), ('что', 'PRON'), ('с', 'ADP'), ('ними', 'PRON'), ('происходит', 'VERB'), (',', 'PUNCT'),
             ('становятся', 'VERB'), ('невротиками', 'NOUN'), (',', 'PUNCT'), ('а', 'CONJ'), ('затем', 'ADV'),
             (',', 'PUNCT'), ('при', 'ADP'), ('прогрессировании', 'NOUN'), ('депрессии', 'NOUN'), (',', 'PUNCT'),
             ('в', 'ADP'), ('той', 'DET'), ('или', 'CONJ'), ('иной', 'DET'), ('степени', 'NOUN'), (',', 'PUNCT'),
             ('и', 'PART'), ('психически', 'ADV'), ('больными', 'ADJ'), ('людьми', 'NOUN'), ('.', 'PUNCT')]
        ]
        fact_result = self.parser.parse(self.test_data_GCRY, [1, 3], "POS")
        self.assertEqual(fact_result, true_result)

    def test_parser_GCRY_all(self):

        true_result = [
            [('Укрепленные', 'ADJ_Nom_Pos_Plur'), ('на', 'ADP'), ('крючьях', 'NOUN_Inan_Loc_Masc_Plur'),
             ('факелы', 'NOUN_Inan_Nom_Masc_Plur'), ('освещали', 'VERB_Ind_Plur_Past_Fin_Act'),
             ('древние', 'ADJ_Acc_Pos_Plur'), ('стены', 'NOUN_Inan_Acc_Fem_Plur'), ('из', 'ADP'),
             ('дикого', 'ADJ_Gen_Pos_Masc_Sing'), ('камня', 'NOUN_Inan_Gen_Masc_Sing'), ('и', 'CONJ'),
             ('тяжелые', 'ADJ_Acc_Pos_Plur'), ('плиты', 'NOUN_Inan_Acc_Fem_Plur'), ('ровного', 'ADJ_Gen_Pos_Masc_Sing'),
             ('пола', 'NOUN_Inan_Gen_Masc_Sing'), ('.', 'PUNCT')],
            [('Те', 'DET_Nom_Plur'), (',', 'PUNCT'), ('кто', 'PRON_Nom_Sing'), ('не', 'PART'),
             ('понимает', 'VERB_Ind_Sing_3_Notpast_Fin_Act'), ('того', 'PRON_Gen_Neut_Sing'), (',', 'PUNCT'),
             ('что', 'PRON_Nom_Neut_Sing'), ('с', 'ADP'), ('ними', 'PRON_Ins_Plur_3'),
             ('происходит', 'VERB_Ind_Sing_3_Notpast_Fin_Act'), (',', 'PUNCT'),
             ('становятся', 'VERB_Ind_Plur_3_Notpast_Fin_Mid'), ('невротиками', 'NOUN_Anim_Ins_Masc_Plur'),
             (',', 'PUNCT'), ('а', 'CONJ'), ('затем', 'ADV_Pos'), (',', 'PUNCT'), ('при', 'ADP'),
             ('прогрессировании', 'NOUN_Inan_Loc_Neut_Sing'), ('депрессии', 'NOUN_Inan_Gen_Fem_Sing'), (',', 'PUNCT'),
             ('в', 'ADP'), ('той', 'DET_Loc_Fem_Sing'), ('или', 'CONJ'), ('иной', 'DET_Loc_Fem_Sing'),
             ('степени', 'NOUN_Inan_Loc_Fem_Sing'), (',', 'PUNCT'), ('и', 'PART'), ('психически', 'ADV_Pos'),
             ('больными', 'ADJ_Ins_Pos_Plur'), ('людьми', 'NOUN_Anim_Ins_Masc_Plur'), ('.', 'PUNCT')]
        ]
        fact_result = self.parser.parse(self.test_data_GCRY, [1, 3, 4], "POS&GRAMMAR")
        self.assertEqual(fact_result, true_result)

    def test_parser_GCRY_person(self):

        true_result = [
            [('Укрепленные', 'O'), ('на', 'O'), ('крючьях', 'O'), ('факелы', 'O'), ('освещали', 'O'), ('древние', 'O'),
             ('стены', 'O'), ('из', 'O'), ('дикого', 'O'), ('камня', 'O'), ('и', 'O'), ('тяжелые', 'O'), ('плиты', 'O'),
             ('ровного', 'O'), ('пола', 'O'), ('.', 'O')],
            [('Те', 'O'), (',', 'O'), ('кто', 'O'), ('не', 'O'),  ('понимает', '3'), ('того', 'O'), (',', 'O'),
             ('что', 'O'), ('с', 'O'), ('ними', '3'), ('происходит', '3'), (',', 'O'), ('становятся', '3'),
             ('невротиками', 'O'), (',', 'O'), ('а', 'O'), ('затем', 'O'), (',', 'O'), ('при', 'O'),
             ('прогрессировании', 'O'), ('депрессии', 'O'), (',', 'O'), ('в', 'O'), ('той', 'O'), ('или', 'O'),
             ('иной', 'O'), ('степени', 'O'), (',', 'O'), ('и', 'O'), ('психически', 'O'), ('больными', 'O'),
             ('людьми', 'O'), ('.', 'O')]
        ]
        fact_result = self.parser.parse(self.test_data_GCRY, [1, 4], "Person")
        self.assertEqual(fact_result, true_result)

    def test_parser_UD_pos(self):

        true_result = [
            [('Анкета', 'NOUN'), ('.', 'PUNCT')],
            [('Начальник', 'NOUN'), ('областного', 'ADJ'), ('управления', 'NOUN'), ('связи', 'NOUN'),
             ('Семен', 'PROPN'), ('Еремеевич', 'PROPN'), ('был', 'AUX'), ('человек', 'NOUN'),
             ('простой', 'ADJ'), (',', 'PUNCT'), ('приходил', 'VERB'), ('на', 'ADP'), ('работу', 'NOUN'),
             ('всегда', 'ADV'), ('вовремя', 'ADV'), (',', 'PUNCT'), ('здоровался', 'VERB'), ('с', 'ADP'),
             ('секретаршей', 'NOUN'), ('за', 'ADP'), ('руку', 'NOUN'), ('и', 'CONJ'), ('иногда', 'ADV'),
             ('даже', 'PART'), ('писал', 'VERB'), ('в', 'ADP'), ('стенгазету', 'NOUN'), ('заметки', 'NOUN'),
             ('под', 'ADP'), ('псевдонимом', 'NOUN'), ('"', 'PUNCT'), ('Муха', 'PROPN'), ('"', 'PUNCT'), ('.', 'PUNCT')]
        ]

        fact_result = self.parser.parse(self.test_data_UD, [1, 3], "POS")
        self.assertEqual(fact_result, true_result)

    def test_parser_UD_all(self):

        true_result = [[('Анкета', 'NOUN_Inan_Nom_Fem_Sing'), ('.', 'PUNCT')],
                       [('Начальник', 'NOUN_Anim_Nom_Masc_Sing'), ('областного', 'ADJ_Gen_Pos_Neut_Sing'),
                        ('управления', 'NOUN_Inan_Gen_Neut_Sing'), ('связи', 'NOUN_Inan_Gen_Fem_Sing'),
                        ('Семен', 'PROPN_Anim_Nom_Masc_Sing'), ('Еремеевич', 'PROPN_Anim_Nom_Masc_Sing'),
                        ('был', 'AUX_Imp_Masc_Ind_Sing_Past_Fin_Act'), ('человек', 'NOUN_Anim_Nom_Masc_Sing'),
                        ('простой', 'ADJ_Nom_Pos_Masc_Sing'), (',', 'PUNCT'),
                        ('приходил', 'VERB_Imp_Masc_Ind_Sing_Past_Fin_Act'), ('на', 'ADP'),
                        ('работу', 'NOUN_Inan_Acc_Fem_Sing'), ('всегда', 'ADV_Pos'), ('вовремя', 'ADV_Pos'),
                        (',', 'PUNCT'), ('здоровался', 'VERB_Imp_Masc_Ind_Sing_Past_Fin_Act'), ('с', 'ADP'),
                        ('секретаршей', 'NOUN_Anim_Ins_Fem_Sing'), ('за', 'ADP'), ('руку', 'NOUN_Inan_Acc_Fem_Sing'),
                        ('и', 'CONJ'), ('иногда', 'ADV_Pos'), ('даже', 'PART'),
                        ('писал', 'VERB_Imp_Masc_Ind_Sing_Past_Fin_Act'), ('в', 'ADP'),
                        ('стенгазету', 'NOUN_Inan_Acc_Fem_Sing'), ('заметки', 'NOUN_Inan_Acc_Fem_Plur'), ('под', 'ADP'),
                        ('псевдонимом', 'NOUN_Inan_Ins_Masc_Sing'), ('"', 'PUNCT'), ('Муха', 'PROPN_Anim_Nom_Fem_Sing'),
                        ('"', 'PUNCT'), ('.', 'PUNCT')]
                       ]
        fact_result = self.parser.parse(self.test_data_UD, [1, 3, 5], "POS&GRAMMAR")
        self.assertEqual(fact_result, true_result)

    def test_parser_UD_case(self):

        true_result = [
            [('Анкета', 'Nom'), ('.', 'O')],
            [('Начальник', 'Nom'), ('областного', 'Gen'), ('управления', 'Gen'), ('связи', 'Gen'), ('Семен', 'Nom'),
             ('Еремеевич', 'Nom'), ('был', 'O'), ('человек', 'Nom'), ('простой', 'Nom'), (',', 'O'), ('приходил', 'O'),
             ('на', 'O'), ('работу', 'Acc'), ('всегда', 'O'), ('вовремя', 'O'), (',', 'O'), ('здоровался', 'O'),
             ('с', 'O'), ('секретаршей', 'Ins'), ('за', 'O'), ('руку', 'Acc'), ('и', 'O'), ('иногда', 'O'), ('даже', 'O'),
             ('писал', 'O'), ('в', 'O'), ('стенгазету', 'Acc'), ('заметки', 'Acc'), ('под', 'O'), ('псевдонимом', 'Ins'),
             ('"', 'O'), ('Муха', 'Nom'), ('"', 'O'), ('.', 'O')]
        ]
        fact_result = self.parser.parse(self.test_data_UD, [1, 5], "Case")
        self.assertEqual(fact_result, true_result)


if __name__ == '__main__':
    unittest.main(verbosity=2)