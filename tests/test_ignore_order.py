import pytest
import re
import datetime
import json
import os
import subprocess
import sys
from unittest import mock
from deepdiff.helper import number_to_string, CannotCompare
from deepdiff import DeepDiff
from decimal import Decimal
from deepdiff.deephash import sha256hex
from tests import CustomClass2


class TestIgnoreOrder:

    @pytest.mark.parametrize("t1, t2, significant_digits, ignore_order, result", [
        (10, 10.0, 5, False, {}),
        ({10: 'a', 11.1: 'b'}, {10.0: 'a', Decimal('11.1000003'): 'b'}, 5, False, {}),
    ])
    def test_type_change_numeric_ignored(self, t1, t2, significant_digits, ignore_order, result):
        ddiff = DeepDiff(t1, t2, ignore_numeric_type_changes=True,
                         significant_digits=significant_digits, ignore_order=ignore_order)
        assert result == ddiff

    @pytest.mark.parametrize("t1, t2, expected_result",
                             [
                                 (10, 10.0, {}),
                                 (10, 10.2, {'values_changed': {'root': {'new_value': 10.2, 'old_value': 10}}}),
                                 (Decimal(10), 10.0, {}),
                                 ({"a": Decimal(10), "b": 12, 11.0: None}, {b"b": 12, "a": 10.0, Decimal(11): None}, {}),
                             ])
    def test_type_change_numeric_when_ignore_order(self, t1, t2, expected_result):
        ddiff = DeepDiff(t1, t2, ignore_order=True, ignore_numeric_type_changes=True, ignore_string_type_changes=True, threshold_to_diff_deeper=0)
        assert expected_result == ddiff

    def test_ignore_order_depth1(self):
        t1 = [{1, 2, 3}, {4, 5}]
        t2 = [{4, 5, 6}, {1, 2, 3}]
        ddiff = DeepDiff(t1, t2, ignore_order=True)
        assert {'set_item_added': ["root[1][6]"]} == ddiff

    def test_ignore_order_depth2(self):
        t1 = [[1, 2, 3], [4, 5]]
        t2 = [[4, 5, 6], [1, 2, 3]]
        ddiff = DeepDiff(t1, t2, ignore_order=True)
        assert {'iterable_item_added': {'root[1][2]': 6}} == ddiff

    def test_ignore_order_depth3(self):
        t1 = [{1, 2, 3}, [{4, 5}]]
        t2 = [[{4, 5, 6}], {1, 2, 3}]
        ddiff = DeepDiff(t1, t2, ignore_order=True)
        assert {'set_item_added': ["root[1][0][6]"]} == ddiff
        assert {"root[1][0][6]"} == ddiff.affected_paths

    def test_ignore_order_depth4(self):
        t1 = [[1, 2, 3, 4], [4, 2, 2, 1]]
        t2 = [[4, 1, 1, 1], [1, 3, 2, 4]]
        ddiff = DeepDiff(t1, t2, ignore_order=True)
        assert {'iterable_item_removed': {'root[1][1]': 2}} == ddiff

    def test_ignore_order_depth5(self):
        t1 = [4, 2, 2, 1]
        t2 = [4, 1, 1, 1]
        ddiff = DeepDiff(t1, t2, ignore_order=True, report_repetition=True, cache_purge_level=0)
        expected = {
            'iterable_item_removed': {
                'root[1]': 2,
                'root[2]': 2
            },
            'repetition_change': {
                'root[3]': {
                    'old_repeat': 1,
                    'new_repeat': 3,
                    'old_indexes': [3],
                    'new_indexes': [1, 2, 3],
                    'value': 1
                }
            }
        }
        assert expected == ddiff
        assert {"root[1]", "root[2]", "root[3]"} == ddiff.affected_paths

        ddiff = DeepDiff(t1, t2, ignore_order=True, report_repetition=False, cache_purge_level=0)
        dist = ddiff._get_rough_distance()
        assert 0.1 == dist

    def test_ignore_order_depth6(self):
        t1 = [[1, 2, 3, 4], [4, 2, 2, 1]]
        t2 = [[4, 1, 1, 1], [1, 3, 2, 4]]
        ddiff = DeepDiff(t1, t2, ignore_order=True, report_repetition=True)
        expected = {
            'iterable_item_removed': {
                'root[1][1]': 2,
                'root[1][2]': 2
            },
            'repetition_change': {
                'root[1][3]': {
                    'old_repeat': 1,
                    'new_repeat': 3,
                    'old_indexes': [3],
                    'new_indexes': [1, 2, 3],
                    'value': 1
                }
            }
        }

        assert expected == ddiff

    def test_list_difference_ignore_order(self):
        t1 = {1: 1, 4: {"a": "hello", "b": [1, 2, 3]}}
        t2 = {1: 1, 4: {"a": "hello", "b": [1, 3, 2, 3]}}
        ddiff = DeepDiff(t1, t2, ignore_order=True)
        assert {} == ddiff

    @pytest.mark.parametrize('t1_0, t2_0', [
        (1, 2),
        (True, False),
        ('a', 'b'),
    ])
    def test_list_difference_of_bool_only_ignore_order(self, t1_0, t2_0):
        t1 = [t1_0]
        t2 = [t2_0]
        ddiff = DeepDiff(t1, t2, ignore_order=True)
        result = {'values_changed': {'root[0]': {'new_value': t2_0, 'old_value': t1_0}}}
        assert result == ddiff

    def test_dictionary_difference_ignore_order(self):
        t1 = {"a": [[{"b": 2, "c": 4}, {"b": 2, "c": 3}]]}
        t2 = {"a": [[{"b": 2, "c": 3}, {"b": 2, "c": 4}]]}
        ddiff = DeepDiff(t1, t2, ignore_order=True)
        assert {} == ddiff
        assert set() == ddiff.affected_paths

    def test_nested_list_ignore_order(self):
        t1 = [1, 2, [3, 4]]
        t2 = [[4, 3, 3], 2, 1]
        ddiff = DeepDiff(t1, t2, ignore_order=True)
        assert {} == ddiff

    def test_nested_list_difference_ignore_order(self):
        t1 = [1, 2, [3, 4]]
        t2 = [[4, 3], 2, 1]
        ddiff = DeepDiff(t1, t2, ignore_order=True)
        assert {} == ddiff

    def test_nested_list_with_dictionarry_difference_ignore_order(self):
        t1 = [1, 2, [3, 4, {1: 2}]]
        t2 = [[4, 3, {1: 2}], 2, 1]

        ddiff = DeepDiff(t1, t2, ignore_order=True)

        result = {}
        assert result == ddiff

    def test_list_difference_ignore_order_report_repetition1(self):
        t1 = [1, 3, 1, 4]
        t2 = [4, 4, 1]
        ddiff = DeepDiff(t1, t2, ignore_order=True, report_repetition=True)
        result = {
            'iterable_item_removed': {
                'root[1]': 3
            },
            'repetition_change': {
                'root[0]': {
                    'old_repeat': 2,
                    'old_indexes': [0, 2],
                    'new_indexes': [2],
                    'value': 1,
                    'new_repeat': 1
                },
                'root[3]': {
                    'old_repeat': 1,
                    'old_indexes': [3],
                    'new_indexes': [0, 1],
                    'value': 4,
                    'new_repeat': 2
                }
            }
        }
        assert result == ddiff

    @pytest.mark.skip
    def test_list_difference_ignore_order_report_repetition2(self):
        t1 = [1, 1, 1]
        t2 = [2, 2]
        ddiff = DeepDiff(t1, t2, ignore_order=True)
        result = {'values_changed': {'root[0]': {'new_value': 2, 'old_value': 1}}}
        assert result == ddiff

        ddiff2 = DeepDiff(t1, t2, ignore_order=True, report_repetition=True, cutoff_intersection_for_pairs=1, cutoff_distance_for_pairs=1)
        result2 = {
            'iterable_item_removed': {
                'root[0]': 1,
                'root[1]': 1,
                'root[2]': 1
            },
            'iterable_item_added': {
                'root[0]': 2,
                'root[1]': 2,
            },
        }
        assert result2 == ddiff2

    @pytest.mark.skip
    def test_list_difference_ignore_order_report_repetition3(self):
        t1 = [{"id": 1}, {"id": 1}, {"id": 1}]
        t2 = [{"id": 1, "name": 1}]

        ddiff2 = DeepDiff(t1, t2, ignore_order=True, report_repetition=True, cutoff_intersection_for_pairs=1, cutoff_distance_for_pairs=1)
        result2 = {
            'iterable_item_removed': {
                'root[1]': {"id": 1},
                'root[2]': {"id": 1},
            },
            'dictionary_item_added': ["root[0]['name']"]
        }
        assert result2 == ddiff2

    @pytest.mark.skip
    def test_list_difference_ignore_order_report_repetition4(self):
        t1 = [{"id": 1}, {"id": 1}, {"id": 1}, {"name": "Joe"}, {"name": "Joe"}]
        t2 = [{"id": 1, "name": 1}, {"id": 1, "name": "Joe"}]

        ddiff2 = DeepDiff(t1, t2, ignore_order=True, report_repetition=True, cutoff_intersection_for_pairs=1, cutoff_distance_for_pairs=1)
        result2 = {
            'iterable_item_removed': {
                'root[2]': {"id": 1},
                'root[3]': {"name": "Joe"},
                'root[4]': {"name": "Joe"},
            },
            'dictionary_item_added': ["root[0]['name']", "root[1]['name']"]
        }
        assert result2 == ddiff2

    def test_nested_list_ignore_order_report_repetition(self):
        t1 = [1, 2, [3, 4]]
        t2 = [[4, 3, 3], 2, 1]
        ddiff = DeepDiff(t1, t2, ignore_order=True, report_repetition=False)
        assert not ddiff

        ddiff2 = DeepDiff(t1, t2, ignore_order=True, report_repetition=True)
        result = {
            'repetition_change': {
                'root[2][0]': {
                    'old_repeat': 1,
                    'new_repeat': 2,
                    'old_indexes': [0],
                    'new_indexes': [1, 2],
                    'value': 3
                }
            }
        }
        assert result == ddiff2
        assert {"root[2][0]"} == ddiff2.affected_paths

    @pytest.mark.skip
    def test_nested_list_and_dict_ignore_order_report_repetition(self):
        """
        This test shows that ignore order is not doing the right thing.

        It should have said that root[1] and root[2] are removed.
        """
        t1 = [{"id": 1}, {"id": 1}, {"id": 1}]
        t2 = [{"id": 1, "name": 1}]
        ddiff = DeepDiff(t1, t2, ignore_order=True)
        result = {'dictionary_item_added': ["root[0]['name']"]}
        assert result == ddiff

        # Here there is nothing that is "repeated" in an iterable
        ddiff2 = DeepDiff(t1, t2, ignore_order=True, report_repetition=True)
        assert result == ddiff2
        assert {"root[2][0]"} == ddiff2.affected_paths

    def test_list_of_unhashable_difference_ignore_order(self):
        t1 = [{"a": 2}, {"b": [3, 4, {1: 1}]}]
        t2 = [{"b": [3, 4, {1: 1}]}, {"a": 2}]
        ddiff = DeepDiff(t1, t2, ignore_order=True)
        assert {} == ddiff

    def test_list_of_unhashable_difference_ignore_order2(self):
        t1 = [1, {"a": 2}, {"b": [3, 4, {1: 1}]}, "B"]
        t2 = [{"b": [3, 4, {1: 1}]}, {"a": 2}, {1: 1}]
        ddiff = DeepDiff(t1, t2, ignore_order=True)
        result = {
            'iterable_item_added': {
                'root[2]': {
                    1: 1
                }
            },
            'iterable_item_removed': {
                'root[3]': 'B',
                'root[0]': 1
            }
        }
        assert result == ddiff

    def test_list_of_unhashable_difference_ignore_order3(self):
        t1 = [1, {"a": 2}, {"a": 2}, {"b": [3, 4, {1: 1}]}, "B"]
        t2 = [{"b": [3, 4, {1: 1}]}, {1: 1}]
        ddiff = DeepDiff(t1, t2, ignore_order=True)
        result = {
            'values_changed': {
                'root[1]': {
                    'new_value': {
                        1: 1
                    },
                    'old_value': {
                        'a': 2
                    }
                }
            },
            'iterable_item_removed': {
                'root[0]': 1,
                'root[4]': 'B'
            }
        }
        assert result == ddiff

    def test_list_of_unhashable_difference_ignore_order_report_repetition(
            self):
        t1 = [1, {"a": 2}, {"a": 2}, {"b": [3, 4, {1: 1}]}, "B"]
        t2 = [{"b": [3, 4, {1: 1}]}, {1: 1}]
        ddiff = DeepDiff(t1, t2, ignore_order=True, report_repetition=True, threshold_to_diff_deeper=0)
        result = {
            'iterable_item_added': {
                'root[1]': {
                    1: 1
                }
            },
            'iterable_item_removed': {
                'root[4]': 'B',
                'root[0]': 1,
                'root[1]': {
                    'a': 2
                },
                'root[2]': {
                    'a': 2
                }
            }
        }
        assert result == ddiff

    def test_list_of_unhashable_difference_ignore_order4(self):
        t1 = [{"a": 2}, {"a": 2}]
        t2 = [{"a": 2}]
        ddiff = DeepDiff(t1, t2, ignore_order=True)
        result = {}
        assert result == ddiff

    def test_list_of_unhashable_difference_ignore_order_report_repetition2(
            self):
        t1 = [{"a": 2}, {"a": 2}]
        t2 = [{"a": 2}]
        ddiff = DeepDiff(t1, t2, ignore_order=True, report_repetition=True)
        result = {
            'repetition_change': {
                'root[0]': {
                    'old_repeat': 2,
                    'new_indexes': [0],
                    'old_indexes': [0, 1],
                    'value': {
                        'a': 2
                    },
                    'new_repeat': 1
                }
            }
        }
        assert result == ddiff

    def test_list_ignore_order_report_repetition(self):
        t1 = [5, 1, 3, 1, 4, 4, 6]
        t2 = [7, 4, 4, 1, 3, 4, 8]
        ddiff = DeepDiff(t1, t2, ignore_order=True, report_repetition=True)
        result = {
            'values_changed': {
                'root[6]': {
                    'new_value': 7,
                    'old_value': 6
                },
                'root[0]': {
                    'new_value': 8,
                    'old_value': 5
                }
            },
            'repetition_change': {
                'root[4]': {
                    'old_repeat': 2,
                    'new_repeat': 3,
                    'old_indexes': [4, 5],
                    'new_indexes': [1, 2, 5],
                    'value': 4
                },
                'root[1]': {
                    'old_repeat': 2,
                    'new_repeat': 1,
                    'old_indexes': [1, 3],
                    'new_indexes': [3],
                    'value': 1
                }
            }
        }
        assert result == ddiff

    def test_list_of_sets_difference_ignore_order(self):
        t1 = [{1}, {2}, {3}]
        t2 = [{4}, {1}, {2}, {3}]
        ddiff = DeepDiff(t1, t2, ignore_order=True)
        result = {'iterable_item_added': {'root[0]': {4}}}
        assert result == ddiff

    def test_list_of_sets_difference_ignore_order_when_there_is_duplicate(
            self):
        t1 = [{1}, {2}, {3}]
        t2 = [{4}, {1}, {2}, {3}, {3}]
        ddiff = DeepDiff(t1, t2, ignore_order=True)
        result = {'iterable_item_added': {'root[0]': {4}}}
        assert result == ddiff

    def test_list_of_sets_difference_ignore_order_when_there_is_duplicate_and_mix_of_hashable_unhashable(
            self):
        t1 = [1, 1, {2}, {3}]
        t2 = [{4}, 1, {2}, {3}, {3}, 1, 1]
        ddiff = DeepDiff(t1, t2, ignore_order=True)
        result = {'iterable_item_added': {'root[0]': {4}}}
        assert result == ddiff

    def test_dictionary_of_list_of_dictionary_ignore_order(self):
        t1 = {
            'item': [{
                'title': 1,
                'http://purl.org/rss/1.0/modules/content/:encoded': '1'
            }, {
                'title': 2,
                'http://purl.org/rss/1.0/modules/content/:encoded': '2'
            }]
        }

        t2 = {
            'item': [{
                'http://purl.org/rss/1.0/modules/content/:encoded': '1',
                'title': 1
            }, {
                'http://purl.org/rss/1.0/modules/content/:encoded': '2',
                'title': 2
            }]
        }

        ddiff = DeepDiff(t1, t2, ignore_order=True)
        assert {} == ddiff

    def test_comprehensive_ignore_order(self):

        t1 = {
            'key1': 'val1',
            'key2': [
                {
                    'key3': 'val3',
                    'key4': 'val4',
                },
                {
                    'key5': 'val5',
                    'key6': 'val6',
                },
            ],
        }

        t2 = {
            'key1': 'val1',
            'key2': [
                {
                    'key5': 'val5',
                    'key6': 'val6',
                },
                {
                    'key3': 'val3',
                    'key4': 'val4',
                },
            ],
        }

        ddiff = DeepDiff(t1, t2, ignore_order=True)
        assert {} == ddiff

    def test_ignore_order_when_objects_similar(self):

        t1 = {
            'key1': 'val1',
            'key2': [
                {
                    'key3': 'val3',
                    'key4': 'val4',
                },
                {
                    'key5': 'val5',
                    'key6': 'val6',
                },
            ],
        }

        t2 = {
            'key1': 'val1',
            'key2': [
                {
                    'key5': 'CHANGE',
                    'key6': 'val6',
                },
                {
                    'key3': 'val3',
                    'key4': 'val4',
                },
            ],
        }

        ddiff = DeepDiff(t1, t2, ignore_order=True)
        assert {'values_changed': {"root['key2'][1]['key5']": {'new_value': 'CHANGE', 'old_value': 'val5'}}} == ddiff

    def test_set_ignore_order_report_repetition(self):
        """
        If this test fails, it means that DeepDiff is not checking
        for set types before general iterables.
        So it forces creating the hashtable because of report_repetition=True.
        """
        t1 = {2, 1, 8}
        t2 = {1, 2, 3, 5}
        ddiff = DeepDiff(t1, t2, ignore_order=True, report_repetition=True)
        result = {
            'set_item_added': {'root[3]', 'root[5]'},
            'set_item_removed': {'root[8]'}
        }
        assert result == ddiff

    def test_custom_objects2(self):
        cc_a = CustomClass2(prop1=["a"], prop2=["b"])
        cc_b = CustomClass2(prop1=["b"], prop2=["b"])
        t1 = [cc_a, CustomClass2(prop1=["c"], prop2=["d"])]
        t2 = [cc_b, CustomClass2(prop1=["c"], prop2=["d"])]

        ddiff = DeepDiff(t1, t2, ignore_order=True)

        result = {'values_changed': {'root[0].prop1[0]': {'new_value': 'b', 'old_value': 'a'}}}
        assert result == ddiff

    def test_custom_object_type_change_when_ignore_order(self):

        class Burrito:
            bread = 'flour'

            def __init__(self):
                self.spicy = True

        class Taco:
            bread = 'flour'

            def __init__(self):
                self.spicy = True

        burrito = Burrito()
        taco = Taco()

        burritos = [burrito]
        tacos = [taco]

        assert not DeepDiff(burritos, tacos, ignore_type_in_groups=[(Taco, Burrito)], ignore_order=True)

    def test_decimal_ignore_order(self):
        t1 = [{1: Decimal('10.1')}, {2: Decimal('10.2')}]
        t2 = [{2: Decimal('10.2')}, {1: Decimal('10.1')}]
        ddiff = DeepDiff(t1, t2, ignore_order=True)
        result = {}
        assert result == ddiff

    @pytest.mark.parametrize('log_scale_similarity_threshold, expected', [
        (
            0.1,
            {}
        ),
        (
            0.01,
            {'values_changed': {'root[1][2]': {'new_value': Decimal('268'), 'old_value': Decimal('290.2')}}}
        ),
    ])
    def test_decimal_log_scale_ignore_order1(self, log_scale_similarity_threshold, expected):
        t1 = [{1: Decimal('10.143')}, {2: Decimal('290.2')}]
        t2 = [{2: Decimal('268')}, {1: Decimal('10.23')}]
        ddiff = DeepDiff(t1, t2, ignore_order=True, use_log_scale=True, log_scale_similarity_threshold=log_scale_similarity_threshold, cutoff_intersection_for_pairs=1)
        assert expected == ddiff

    @pytest.mark.parametrize("t1, t2, significant_digits, ignore_order", [
        (100000, 100021, 3, False),
        ([10, 12, 100000], [50, 63, 100021], 3, False),
        ([10, 12, 100000], [50, 63, 100021], 3, True),
    ])
    def test_number_to_string_func(self, t1, t2, significant_digits, ignore_order):
        def custom_number_to_string(number, *args, **kwargs):
            number = 100 if number < 100 else number
            return number_to_string(number, *args, **kwargs)

        ddiff = DeepDiff(t1, t2, significant_digits=3, number_format_notation="e",
                         number_to_string_func=custom_number_to_string)

        assert {} == ddiff

    def test_ignore_type_in_groups_numbers_and_strings_when_ignore_order(self):
        t1 = [1, 2, 3, 'a']
        t2 = [1.0, 2.0, 3.3, b'a']
        ddiff = DeepDiff(t1, t2, ignore_numeric_type_changes=True, ignore_string_type_changes=True, ignore_order=True)
        result = {'values_changed': {'root[2]': {'new_value': 3.3, 'old_value': 3}}}
        assert result == ddiff

    def test_ignore_string_type_changes_when_dict_keys_merge_is_not_deterministic(self):
        t1 = {'a': 10, b'a': 20}
        t2 = {'a': 11, b'a': 22}
        ddiff = DeepDiff(t1, t2, ignore_numeric_type_changes=True, ignore_string_type_changes=True, ignore_order=True)
        result = {'values_changed': {"root['a']": {'new_value': 22, 'old_value': 20}}}
        alternative_result = {'values_changed': {"root['a']": {'new_value': 11, 'old_value': 10}}}
        assert result == ddiff or alternative_result == ddiff

    def test_skip_exclude_path5(self):
        exclude_paths = ["root[0]['e']", "root[1]['e']"]

        t1 = [{'a': 1, 'b': 'randomString', 'e': "1111"}]
        t2 = [{'a': 1, 'b': 'randomString', 'e': "2222"}]

        ddiff = DeepDiff(t1, t2, exclude_paths=exclude_paths,
                         ignore_order=True, report_repetition=False)
        result = {}
        assert result == ddiff

    def test_skip_str_type_in_dict_on_list_when_ignored_order(self):
        t1 = [{1: "a"}]
        t2 = [{}]
        ddiff = DeepDiff(t1, t2, exclude_types=[str], ignore_order=True)
        result = {}
        assert result == ddiff

    @mock.patch('deepdiff.diff.logger')
    @mock.patch('deepdiff.diff.DeepHash')
    def test_diff_when_hash_fails(self, mock_DeepHash, mock_logger):
        mock_DeepHash.side_effect = ValueError('Boom!')
        t1 = {"blah": {4}, 2: 1337}
        t2 = {"blah": {4}, 2: 1337}
        with pytest.raises(ValueError) as exp:
            DeepDiff(t1, t2, ignore_order=True)
        assert 'Boom!' == str(exp.value)

    def test_bool_vs_number(self):
        t1 = {
            "A List": [
                {
                    "Value One": True,
                    "Value Two": 1
                }
            ],
        }

        t2 = {
            "A List": [
                {
                    "Value Two": 1,
                    "Value One": True
                }
            ],
        }

        ddiff = DeepDiff(t1, t2, ignore_order=True, cutoff_intersection_for_pairs=1)
        assert ddiff == {}

    @pytest.mark.parametrize('max_passes, expected', [
        (0, {'values_changed': {'root[0]': {'new_value': {'key5': 'CHANGE', 'key6': 'val6'}, 'old_value': {'key3': [[[[[1, 2, 4, 5]]]]], 'key4': [7, 8]}}, 'root[1]': {'new_value': {'key3': [[[[[1, 3, 5, 4]]]]], 'key4': [7, 8]}, 'old_value': {'key5': 'val5', 'key6': 'val6'}}}}),
        (1, {'values_changed': {"root[1]['key5']": {'new_value': 'CHANGE', 'old_value': 'val5', 'new_path': "root[0]['key5']"}, "root[0]['key3'][0]": {'new_value': [[[[1, 3, 5, 4]]]], 'old_value': [[[[1, 2, 4, 5]]]], 'new_path': "root[1]['key3'][0]"}}}),
        (22, {'values_changed': {"root[1]['key5']": {'new_value': 'CHANGE', 'old_value': 'val5', 'new_path': "root[0]['key5']"}, "root[0]['key3'][0][0][0][0][1]": {'new_value': 3, 'old_value': 2, 'new_path': "root[1]['key3'][0][0][0][0][1]"}}})
    ])
    def test_ignore_order_max_passes(self, max_passes, expected):
        t1 = [
            {
                'key3': [[[[[1, 2, 4, 5]]]]],
                'key4': [7, 8],
            },
            {
                'key5': 'val5',
                'key6': 'val6',
            },
        ]

        t2 = [
            {
                'key5': 'CHANGE',
                'key6': 'val6',
            },
            {
                'key3': [[[[[1, 3, 5, 4]]]]],
                'key4': [7, 8],
            },
        ]

        ddiff = DeepDiff(t1, t2, ignore_order=True, max_passes=max_passes, verbose_level=2, cache_size=5000, cutoff_intersection_for_pairs=1, threshold_to_diff_deeper=0)
        assert expected == ddiff

    @pytest.mark.parametrize('max_diffs, expected', [
        (1, {}),
        (65, {'values_changed': {"root[1]['key5']": {'new_value': 'CHANGE', 'old_value': 'val5', 'new_path': "root[0]['key5']"}}}),
        (80, {'values_changed': {"root[1]['key5']": {'new_value': 'CHANGE', 'old_value': 'val5', 'new_path': "root[0]['key5']"}, "root[0]['key3'][0][0][0][0][1]": {'new_value': 3, 'old_value': 2, 'new_path': "root[1]['key3'][0][0][0][0][1]"}}}),
    ])
    def test_ignore_order_max_diffs(self, max_diffs, expected):
        t1 = [
            {
                'key3': [[[[[1, 2, 4, 5]]]]],
                'key4': [7, 8],
            },
            {
                'key5': 'val5',
                'key6': 'val6',
            },
        ]

        t2 = [
            {
                'key5': 'CHANGE',
                'key6': 'val6',
            },
            {
                'key3': [[[[[1, 3, 5, 4]]]]],
                'key4': [7, 8],
            },
        ]

        # Note: these tests are not exactly deterministic
        ddiff = DeepDiff(t1, t2, ignore_order=True, max_diffs=max_diffs, verbose_level=2, cache_size=5000, cutoff_intersection_for_pairs=1)
        assert expected == ddiff

    def test_stats_that_include_distance_cache_hits(self):
        t1 = [
            [1, 2, 3, 9], [9, 8, 5, 9]
        ]

        t2 = [
            [1, 2, 4, 10], [4, 2, 5]
        ]

        diff = DeepDiff(t1, t2, ignore_order=True, cache_size=5000, cutoff_intersection_for_pairs=1)
        expected = {
            'PASSES COUNT': 7,
            'DIFF COUNT': 37,
            'DISTANCE CACHE HIT COUNT': 0,
            'MAX PASS LIMIT REACHED': False,
            'MAX DIFF LIMIT REACHED': False,
            # Phase 4 (multiprocessing-aware stats) added zeroed worker keys to
            # every run; serial diffs have nothing to aggregate so they stay 0.
            'WORKER DIFF COUNT': 0,
            'WORKER PASSES COUNT': 0,
            'WORKER DISTANCE CACHE HIT COUNT': 0,
            'WORKER BATCH COUNT': 0,
        }
        assert expected == diff.get_stats()

    def test_ignore_order_report_repetition_and_self_loop(self):
        t1 = [[1, 2, 1, 3]]
        t1.append(t1)

        t2 = [[1, 2, 2, 2, 4]]
        t2.append(t2)

        diff = DeepDiff(t1, t2, ignore_order=True, cutoff_intersection_for_pairs=1)
        expected = {
            'values_changed': {
                'root[0][3]': {
                    'new_value': 4,
                    'old_value': 3
                },
                'root[1]': {
                    'new_value': t2,
                    'old_value': t1
                }
            }
        }
        assert expected == diff

        diff2 = DeepDiff(t1, t2, ignore_order=True, cache_size=5000, cutoff_intersection_for_pairs=1)
        assert expected == diff2

    def test_ignore_order_with_sha256_hash(self):
        t1 = [
            [1, 2, 3, 9], [9, 8, 5, 9]
        ]

        t2 = [
            [1, 2, 3, 10], [8, 2, 5]
        ]
        diff = DeepDiff(t1, t2, ignore_order=True, hasher=sha256hex, cutoff_intersection_for_pairs=1)
        expected = {
            'values_changed': {
                'root[0][3]': {
                    'new_value': 10,
                    'old_value': 9
                },
                'root[1][0]': {
                    'new_value': 2,
                    'old_value': 9
                }
            }
        }
        assert expected == diff

    def test_ignore_order_cache_for_individual_distances(self):
        t1 = [[1, 2, 'B', 3], 'B']
        t2 = [[1, 2, 3, 5], 5]
        diff = DeepDiff(t1, t2, ignore_order=True, cache_size=5000, cutoff_intersection_for_pairs=1)
        expected = {
            'values_changed': {
                'root[1]': {
                    'new_value': 5,
                    'old_value': 'B'
                }
            },
            'iterable_item_added': {
                'root[0][3]': 5
            },
            'iterable_item_removed': {
                'root[0][2]': 'B'
            }
        }
        assert expected == diff

        stats = diff.get_stats()
        expected_stats = {
            'PASSES COUNT': 3,
            'DIFF COUNT': 13,
            'DISTANCE CACHE HIT COUNT': 1,
            'MAX PASS LIMIT REACHED': False,
            'MAX DIFF LIMIT REACHED': False,
            # Phase 4: zeroed worker aggregates always present in get_stats().
            'WORKER DIFF COUNT': 0,
            'WORKER PASSES COUNT': 0,
            'WORKER DISTANCE CACHE HIT COUNT': 0,
            'WORKER BATCH COUNT': 0,
        }
        assert expected_stats == stats

        t1 = [[1, 2, 'B', 3], 5]
        t2 = [[1, 2, 3, 5], 'B']
        diff2 = DeepDiff(t1, t2, ignore_order=True, cache_size=5000, cutoff_intersection_for_pairs=1)
        assert expected_stats == diff2.get_stats()

    def test_cutoff_distance_for_pairs(self):
        t1 = [[1.0]]
        t2 = [[20.0]]
        diff1 = DeepDiff(t1, t2, ignore_order=True, cutoff_distance_for_pairs=0.3)
        expected1 = {'values_changed': {'root[0][0]': {'new_value': 20.0, 'old_value': 1.0}}}
        assert expected1 == diff1

        diff2 = DeepDiff(t1, t2, ignore_order=True, cutoff_distance_for_pairs=0.1)
        expected2 = {'values_changed': {'root[0]': {'new_value': [20.0], 'old_value': [1.0]}}}
        assert expected2 == diff2

        diff_with_dist = DeepDiff(1.0, 20.0, get_deep_distance=True)
        expected = {'values_changed': {'root': {'new_value': 20.0, 'old_value': 1.0}}, 'deep_distance': 0.2714285714285714}

        assert expected == diff_with_dist

    def test_ignore_order_and_group_by1(self):
        t1 = [
            {'id': 'AA', 'name': 'Joe', 'ate': ['Nothing']},
            {'id': 'BB', 'name': 'James', 'ate': ['Chips', 'Cheese']},
            {'id': 'CC', 'name': 'Mike', 'ate': ['Apple']},
        ]

        t2 = [
            {'id': 'BB', 'name': 'James', 'ate': ['Chips', 'Brownies', 'Cheese']},
            {'id': 'AA', 'name': 'Joe', 'ate': ['Nothing']},
            {'id': 'CC', 'name': 'Mike', 'ate': ['Apple', 'Apple']},
        ]

        diff = DeepDiff(t1, t2, group_by='id', ignore_order=False)
        expected = {'iterable_item_added': {"root['BB']['ate'][1]": 'Brownies', "root['CC']['ate'][1]": 'Apple'}}
        assert expected == diff

        diff2 = DeepDiff(t1, t2, group_by='id', ignore_order=True)
        expected2 = {'iterable_item_added': {"root['BB']['ate'][1]": 'Brownies'}}
        assert expected2 == diff2

    def test_ignore_order_and_group_by2(self):
        t1_data = [{'id': '1', 'codes': ['1', '2', '3']}]
        t2_data = [{'id': '1', 'codes': ['1', '2', '4']}]
        diff = DeepDiff(t1_data, t2_data, group_by='id', ignore_order=True)
        expected = {'values_changed': {"root['1']['codes'][2]": {'new_value': '4', 'old_value': '3'}}}
        assert expected == diff

    def test_ignore_order_and_group_by3(self):
        t1 = [{
            'id':
            '5ec52e',
            'products': [{
                'lineNumber': 1,
                'productPrice': '2.39',
                'productQuantity': 2
            }, {
                'lineNumber': 2,
                'productPrice': '4.44',
                'productQuantity': 1
            }],
        }]

        t2 = [{
            'id':
            '5ec52e',
            'products': [
                {
                    'lineNumber': 2,
                    'productPrice': '4.44',
                    'productQuantity': 1
                },
                {
                    'lineNumber': 1,
                    'productPrice': '2.39',
                    'productQuantity': 2
                },
            ],
        }]

        diff = DeepDiff(t1, t2, group_by='id', ignore_order=True)
        assert {} == diff

    def test_ignore_order_and_group_by4(self):
        t1 = [
            {
                "id": "1",
                "field_01": {
                    "subfield_01": {
                        "subfield_02": {"subfield_03": "1"},
                    }
                },
            },
            {"id": "2", "field_01": ["1", "2", "3"]},
            {"id": "3", "field_01": ["1", "2", "3"]},
        ]
        t2 = [
            {
                "id": "1",
                "field_01": {
                    "subfield_01": {
                        "subfield_02": {"subfield_03": "2"},
                    }
                },
            },
            {"id": "2", "field_01": ["4", "5", "6"]},
            {"id": "3", "field_01": ["7", "8", "9"]},
        ]
        diff = DeepDiff(t1, t2, group_by='id', ignore_order=True)
        expected = {
            'values_changed': {
                "root['1']['field_01']['subfield_01']['subfield_02']['subfield_03']": {
                    'new_value': '2',
                    'old_value': '1'
                },
                "root['2']['field_01'][1]": {
                    'new_value': '5',
                    'old_value': '2'
                },
                "root['3']['field_01'][2]": {
                    'new_value': '9',
                    'old_value': '3'
                },
                "root['2']['field_01'][0]": {
                    'new_value': '4',
                    'old_value': '1'
                },
                "root['3']['field_01'][1]": {
                    'new_value': '8',
                    'old_value': '2'
                },
                "root['3']['field_01'][0]": {
                    'new_value': '7',
                    'old_value': '1'
                },
                "root['2']['field_01'][2]": {
                    'new_value': '6',
                    'old_value': '3'
                }
            }
        }

        assert expected == diff

    def test_math_epsilon_when_ignore_order_in_dictionary(self):
        a = {'x': 0.001}
        b = {'x': 0.0011}
        diff = DeepDiff(a, b, ignore_order=True)
        assert {'values_changed': {"root['x']": {'new_value': 0.0011, 'old_value': 0.001}}} == diff

        diff2 = DeepDiff(a, b, ignore_order=True, math_epsilon=0.01)
        assert {} == diff2

    def test_math_epsilon_when_ignore_order_in_list(self):
        a = [0.001, 2]
        b = [2, 0.0011]
        diff = DeepDiff(a, b, ignore_order=True)
        assert {'values_changed': {'root[0]': {'new_value': 0.0011, 'old_value': 0.001}}} == diff

        diff2 = DeepDiff(a, b, ignore_order=True, math_epsilon=0.01)
        assert {} == diff2

    def test_math_epsilon_when_ignore_order_in_nested_list(self):
        a = [{'x': 0.001}, {'y': 2.00002}]
        b = [{'x': 0.0011}, {'y': 2}]

        diff = DeepDiff(a, b, ignore_order=True, math_epsilon=0.01)
        expected = {'values_changed': {'root[0]': {'new_value': {'x': 0.0011}, 'old_value': {'x': 0.001}}, 'root[1]': {'new_value': {'y': 2}, 'old_value': {'y': 2.00002}}}}
        assert expected == diff

    def test_datetime_and_ignore_order(self):
        diff = DeepDiff(
            [{'due_date': datetime.date(2024, 2, 1)}],
            [{'due_date': datetime.date(2024, 2, 2)}],
            ignore_order=True,
            ignore_numeric_type_changes=True
        )
        assert {} != diff



class TestCompareFuncIgnoreOrder:

    def test_ignore_order_with_compare_func_to_guide_comparison(self):
        t1 = [
            {
                'id': 1,
                'value': [1]
            },
            {
                'id': 2,
                'value': [7, 8, 1]
            },
            {
                'id': 3,
                'value': [7, 8],
            },
        ]

        t2 = [
            {
                'id': 2,
                'value': [7, 8]
            },
            {
                'id': 3,
                'value': [7, 8, 1],
            },
            {
                'id': 1,
                'value': [1]
            },
        ]

        expected = {
            'values_changed': {
                "root[2]['id']": {
                    'new_value': 2,
                    'old_value': 3
                },
                "root[1]['id']": {
                    'new_value': 3,
                    'old_value': 2
                }
            }
        }

        expected_with_compare_func = {
            'iterable_item_added': {
                "root[2]['value'][2]": 1
            },
            'iterable_item_removed': {
                "root[1]['value'][2]": 1
            }
        }

        ddiff = DeepDiff(t1, t2, ignore_order=True)

        assert expected == ddiff

        def compare_func(x, y, level=None):
            try:
                return x['id'] == y['id']
            except Exception:
                raise CannotCompare() from None

        ddiff2 = DeepDiff(t1, t2, ignore_order=True, iterable_compare_func=compare_func)
        assert expected_with_compare_func == ddiff2
        assert ddiff != ddiff2

        ddiff3 = DeepDiff(t1, t2, ignore_order=True, iterable_compare_func=compare_func, view='tree')
        assert 1 == ddiff3['iterable_item_removed'][0].t1
        assert 1 == ddiff3['iterable_item_added'][0].t2

    def test_ignore_order_with_compare_func_can_throw_cannot_compare(self):
        t1 = [
            {1},
            {
                'id': 2,
                'value': [7, 8, 1]
            },
            {
                'id': 3,
                'value': [7, 8],
            },
        ]

        t2 = [
            {
                'id': 2,
                'value': [7, 8]
            },
            {
                'id': 3,
                'value': [7, 8, 1],
            },
            {},
        ]

        expected = {
            'type_changes': {
                'root[0]': {
                    'old_type': set,
                    'new_type': dict,
                    'old_value': {1},
                    'new_value': {}
                }
            },
            'values_changed': {
                "root[2]['id']": {
                    'new_value': 2,
                    'old_value': 3
                },
                "root[1]['id']": {
                    'new_value': 3,
                    'old_value': 2
                }
            }
        }
        expected_with_compare_func = {
            'type_changes': {
                'root[0]': {
                    'old_type': set,
                    'new_type': dict,
                    'old_value': {1},
                    'new_value': {}
                }
            },
            'iterable_item_added': {
                "root[2]['value'][2]": 1
            },
            'iterable_item_removed': {
                "root[1]['value'][2]": 1
            }
        }

        ddiff = DeepDiff(t1, t2, cutoff_intersection_for_pairs=1, cutoff_distance_for_pairs=1, ignore_order=True, threshold_to_diff_deeper=0)
        assert expected == ddiff

        def compare_func(x, y, level=None):
            try:
                return x['id'] == y['id']
            except Exception:
                raise CannotCompare() from None

        ddiff2 = DeepDiff(t1, t2, ignore_order=True, cutoff_intersection_for_pairs=1, cutoff_distance_for_pairs=1, iterable_compare_func=compare_func, threshold_to_diff_deeper=0)
        assert expected_with_compare_func == ddiff2
        assert ddiff != ddiff2

    def test_ignore_order_with_compare_func_with_one_each_hashes_added_hashes_removed(self):
        """
        Scenario:
        In this example which demonstrates the problem... We have two dictionaries containing lists for
        individualNames.  Each list contains exactly 2 elements. The effective change is that we are
        replacing the 2nd element in the list.
        NOTE: This is considered a REPLACEMENT of the second element and not an UPDATE of the element
        because we are providing a custom compare_func which will determine matching elements based on
        the value of the nameIdentifier field.  If the custom compare_func is not used, then
        deepdiff.diff will mistakenly treat the difference as being individual field updates for every
        field in the second element of the list.

        Intent:
        Use our custom compare_func, since we have provided it.
        We need to fall into self._precalculate_distance_by_custom_compare_func
        To do this, we are proposing a change to deepdiff.diff line 1128:

        Original:
            if hashes_added and hashes_removed and self.iterable_compare_func and len(hashes_added) > 1 and len(hashes_removed) > 1:

        Proposed/Updated:
            if hashes_added and hashes_removed \
            and self.iterable_compare_func \
            and len(hashes_added) > 0 and len(hashes_removed) > 0:

        NOTE: It is worth mentioning that deepdiff.diff line 1121, might also benefit by changing the length conditions
        to evaluate for > 0 (rather than > 1).
        """

        t1 = {
            "individualNames": [
                {
                    "firstName": "Johnathan",
                    "lastName": "Doe",
                    "prefix": "COLONEL",
                    "middleName": "A",
                    "primaryIndicator": True,
                    "professionalDesignation": "PHD",
                    "suffix": "SR",
                    "nameIdentifier": "00001"
                },
                {
                    "firstName": "John",
                    "lastName": "Doe",
                    "prefix": "",
                    "middleName": "",
                    "primaryIndicator": False,
                    "professionalDesignation": "",
                    "suffix": "SR",
                    "nameIdentifier": "00002"
                }
            ]
        }

        t2 = {
            "individualNames": [
                {
                    "firstName": "Johnathan",
                    "lastName": "Doe",
                    "prefix": "COLONEL",
                    "middleName": "A",
                    "primaryIndicator": True,
                    "professionalDesignation": "PHD",
                    "suffix": "SR",
                    "nameIdentifier": "00001"
                },
                {
                    "firstName": "Johnny",
                    "lastName": "Doe",
                    "prefix": "",
                    "middleName": "A",
                    "primaryIndicator": False,
                    "professionalDesignation": "",
                    "suffix": "SR",
                    "nameIdentifier": "00003"
                }
            ]
        }
        def compare_func(item1, item2, level=None):
            print("*** inside compare ***")
            it1_keys = item1.keys()

            try:

                # --- individualNames ---
                if 'nameIdentifier' in it1_keys and 'lastName' in it1_keys:
                    match_result = item1['nameIdentifier'] == item2['nameIdentifier']
                    print("individualNames - matching result:", match_result)
                    return match_result
                else:
                    print("Unknown list item...", "matching result:", item1 == item2)
                    return item1 == item2
            except Exception:
                raise CannotCompare() from None
        # ---------------------------- End of nested function

        actual_diff = DeepDiff(t1, t2, report_repetition=True,
                             ignore_order=True, iterable_compare_func=compare_func, cutoff_intersection_for_pairs=1)

        old_invalid_diff = {
            'values_changed': {"root['individualNames'][1]['firstName']": {'new_value': 'Johnny', 'old_value': 'John'},
                               "root['individualNames'][1]['middleName']": {'new_value': 'A', 'old_value': ''},
                               "root['individualNames'][1]['nameIdentifier']": {'new_value': '00003',
                                                                                'old_value': '00002'}}}
        new_expected_diff = {'iterable_item_added': {
            "root['individualNames'][1]": {'firstName': 'Johnny', 'lastName': 'Doe', 'prefix': '', 'middleName': 'A',
                                           'primaryIndicator': False, 'professionalDesignation': '', 'suffix': 'SR',
                                           'nameIdentifier': '00003'}}, 'iterable_item_removed': {
            "root['individualNames'][1]": {'firstName': 'John', 'lastName': 'Doe', 'prefix': '', 'middleName': '',
                                           'primaryIndicator': False, 'professionalDesignation': '', 'suffix': 'SR',
                                           'nameIdentifier': '00002'}}}

        assert old_invalid_diff != actual_diff
        assert new_expected_diff == actual_diff


class TestDynamicIgnoreOrder:
    def test_ignore_order_func(self):
        t1 = {
            "order_matters": [
                {1},
                {
                    'id': 2,
                    'value': [7, 8, 1]
                },
                {
                    'id': 3,
                    'value': [7, 8],
                },
            ],
            "order_does_not_matter": [
                {1},
                {
                    'id': 2,
                    'value': [7, 8, 1]
                },
                {
                    'id': 3,
                    'value': [7, 8],
                },
            ]
        }

        t2 = {
            "order_matters": [
                {
                    'id': 2,
                    'value': [7, 8]
                },
                {
                    'id': 3,
                    'value': [7, 8, 1],
                },
                {},
            ],
            "order_does_not_matter": [
                {
                    'id': 2,
                    'value': [7, 8]
                },
                {
                    'id': 3,
                    'value': [7, 8, 1],
                },
                {},
            ]
        }

        def ignore_order_func(level):
            return "order_does_not_matter" in level.path()

        ddiff = DeepDiff(t1, t2, cutoff_intersection_for_pairs=1, cutoff_distance_for_pairs=1, ignore_order_func=ignore_order_func, threshold_to_diff_deeper=0)

        expected = {
            'type_changes': {
                "root['order_matters'][0]": {
                    'old_type': set,
                    'new_type': dict,
                    'old_value': {1},
                    'new_value': {'id': 2, 'value': [7, 8]}
                },
                "root['order_does_not_matter'][0]": {
                    'old_type': set,
                    'new_type': dict,
                    'old_value': {1},
                    'new_value': {}
                }
            },
            'dictionary_item_removed': [
                "root['order_matters'][2]['id']",
                "root['order_matters'][2]['value']"
            ],
            'values_changed': {
                "root['order_matters'][1]['id']": {'new_value': 3, 'old_value': 2},
                "root['order_does_not_matter'][2]['id']": {'new_value': 2, 'old_value': 3},
                "root['order_does_not_matter'][1]['id']": {'new_value': 3, 'old_value': 2}
            }
        }
        assert expected == ddiff


class TestDecodingErrorIgnoreOrder:

    EXPECTED_MESSAGE1 = (
        "'utf-8' codec can't decode byte 0xc3 in position 0: Can not produce a hash for root: invalid continuation byte in '('. "
        "Please either pass ignore_encoding_errors=True or pass the encoding via encodings=['utf-8', '...'].")

    EXPECTED_MESSAGE2 = (
        "'utf-8' codec can't decode byte 0xbc in position 0: Can not produce a hash for root: invalid start byte in ' cup of flour'. "
        "Please either pass ignore_encoding_errors=True or pass the encoding via encodings=['utf-8', '...'].")

    @pytest.mark.parametrize('test_num, item, encodings, ignore_encoding_errors, expected_result, expected_message', [
        (1, b'\xc3\x28', None, False, UnicodeDecodeError, EXPECTED_MESSAGE1),
        (2, b'\xc3\x28', ['utf-8'], False, UnicodeDecodeError, EXPECTED_MESSAGE1),
        (3, b'\xc3\x28', ['utf-8'], True, {'values_changed': {'root[0]': {'new_value': b'\xc3(', 'old_value': b'foo'}}}, None),
        (4, b"\xbc cup of flour", ['utf-8'], False, UnicodeDecodeError, EXPECTED_MESSAGE2),
        (5, b"\xbc cup of flour", ['utf-8'], True, {'values_changed': {'root[0]': {'new_value': b'\xbc cup of flour', 'old_value': b'foo'}}}, None),
        (6, b"\xbc cup of flour", ['utf-8', 'latin-1'], False, {'values_changed': {'root[0]': {'new_value': b'\xbc cup of flour', 'old_value': b'foo'}}}, None),
    ])
    @mock.patch('deepdiff.diff.logger')
    def test_diff_encodings(self, mock_logger, test_num, item, encodings, ignore_encoding_errors, expected_result, expected_message):
        if UnicodeDecodeError == expected_result:
            with pytest.raises(expected_result) as exc_info:
                DeepDiff([b'foo'], [item], encodings=encodings, ignore_encoding_errors=ignore_encoding_errors, ignore_order=True)
            assert expected_message == str(exc_info.value), f"test_diff_encodings test #{test_num} failed."
        else:
            result = DeepDiff([b'foo'], [item], encodings=encodings, ignore_encoding_errors=ignore_encoding_errors, ignore_order=True)
            assert expected_result == result, f"test_diff_encodings test #{test_num} failed."


class TestErrorMessagesWhenIgnoreOrder:

    @mock.patch('deepdiff.diff.logger')
    def test_error_messages_when_ignore_order(self, mock_logger):
        t1 = {'x': 0, 'y': [0, 'a', 'b', 'c']}
        t2 = {'x': 1, 'y': [1, 'c', 'b', 'a']}

        exclude = [re.compile(r"\['x'\]"), re.compile(r"\['y'\]\[0\]")]

        result = DeepDiff(t1, t2, ignore_order=True, exclude_regex_paths=exclude)
        assert {} == result

        assert not mock_logger.error.called



class TestIgnoreOrderNumericTypeChange:
    """Regression tests for GitHub issue #485.

    When ignore_order=True, numerically-equal values of different numeric types
    (e.g. int 1 vs float 1.0) must still be reported as type_changes.

    Root cause: Python's hash equality (hash(1) == hash(1.0)) and value equality
    (1 == 1.0) caused both items to land in the same DeepHash bucket, so they
    were treated as identical and silently dropped from the diff result.
    """

    def test_int_vs_float_in_list_of_dicts(self):
        """Core regression: type change inside a dict nested in a list."""
        result = DeepDiff([{"a": 1}], [{"a": 1.0}], ignore_order=True)
        assert "type_changes" in result, (
            "Expected type_changes between int 1 and float 1.0, got: %s" % result
        )
        assert result["type_changes"]["root[0]['a']"]["old_type"] is int
        assert result["type_changes"]["root[0]['a']"]["new_type"] is float

    def test_ignore_numeric_type_changes_suppresses_report(self):
        """When ignore_numeric_type_changes=True the type change must be hidden."""
        result = DeepDiff(
            [{"a": 1}], [{"a": 1.0}],
            ignore_order=True,
            ignore_numeric_type_changes=True,
        )
        assert result == {}, (
            "With ignore_numeric_type_changes=True there should be no diff, got: %s" % result
        )

    def test_value_change_still_detected(self):
        """Ordinary value differences must still be detected."""
        result = DeepDiff([1], [2], ignore_order=True)
        assert result != {}, "Expected a diff between [1] and [2]"

    def test_reorder_no_false_positive(self):
        """A simple reorder of identical values must not trigger type_changes."""
        result = DeepDiff([1, 2, 3], [3, 2, 1], ignore_order=True)
        assert result == {}, "Reordering identical ints must not produce a diff"

    def test_mixed_list_one_type_change(self):
        """Only the item with a type change should appear in the diff."""
        result = DeepDiff(
            [{"a": 1}, {"b": 2}],
            [{"b": 2}, {"a": 1.0}],
            ignore_order=True,
        )
        assert "type_changes" in result
        assert "root[0]['a']" in result["type_changes"]

    def test_ignore_order_false_unchanged(self):
        """The ignore_order=False path must continue to work as before."""
        result = DeepDiff([{"a": 1}], [{"a": 1.0}], ignore_order=False)
        assert "type_changes" in result


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Self-contained scripts run in subprocesses with different PYTHONHASHSEED
# values. Config-audit pipelines run DeepDiff in fresh processes, so the
# serialized result must be byte-identical no matter how the interpreter
# happens to randomize hash seeds.
_LIST_OF_DICTS_SCRIPT = r'''
import sys
from deepdiff import DeepDiff

t1 = {
    "services": [
        {"name": "web", "port": 80, "tags": ["http", "public"]},
        {"name": "api", "port": 8080, "tags": ["http", "internal"]},
        {"name": "db", "port": 5432, "tags": ["tcp"]},
    ],
    "features": {"auth": True, "metrics": False},
}
t2 = {
    "services": [
        {"name": "db", "port": 5433, "tags": ["tcp"]},
        {"name": "api", "port": 8081, "tags": ["internal", "http"]},
        {"name": "web", "port": 81, "tags": ["public", "http"]},
    ],
    "features": {"metrics": False, "auth": True},
}
sys.stdout.write(DeepDiff(t1, t2, ignore_order=True).to_json())
'''

_SET_SCRIPT = r'''
import sys
from deepdiff import DeepDiff

t1 = {"tags": {"alpha", "beta", "gamma", "delta"}}
t2 = {"tags": {"beta", "gamma", "delta", "epsilon", "zeta"}}
sys.stdout.write(DeepDiff(t1, t2, ignore_order=True).to_json())
'''


def _run_script_with_hash_seeds(script, seeds):
    outputs = []
    for seed in seeds:
        env = dict(os.environ, PYTHONHASHSEED=str(seed))
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, env=env, cwd=REPO_ROOT,
        )
        assert proc.returncode == 0, proc.stderr
        outputs.append(proc.stdout)
    return outputs


class TestIgnoreOrderDeterminism:
    """Pin down stable, reproducible ignore_order results for config audit.

    The same inputs must always produce the same report: no extra entries,
    no missing entries, and byte-identical serialization across runs and
    across processes.
    """

    NESTED_T1 = {
        "services": [
            {"name": "web", "port": 80, "tags": ["http", "public"],
             "limits": {"cpu": "500m", "mem": "256Mi"}},
            {"name": "api", "port": 8080, "tags": ["http", "internal"],
             "limits": {"cpu": "1", "mem": "512Mi"}},
            {"name": "db", "port": 5432, "tags": ["tcp"],
             "limits": {"cpu": "2", "mem": "1Gi"}},
        ],
        "features": {"auth": True, "metrics": False},
    }
    NESTED_T2 = {
        "services": [
            {"name": "db", "port": 5432, "tags": ["tcp"],
             "limits": {"cpu": "2", "mem": "1Gi"}},
            {"name": "api", "port": 8080, "tags": ["internal", "http"],
             "limits": {"cpu": "2", "mem": "512Mi"}},
            {"name": "web", "port": 80, "tags": ["public", "http"],
             "limits": {"cpu": "500m", "mem": "256Mi"}},
        ],
        "features": {"metrics": False, "auth": True},
    }

    def test_repeated_runs_produce_byte_identical_json(self):
        outputs = {
            DeepDiff(self.NESTED_T1, self.NESTED_T2, ignore_order=True).to_json()
            for _ in range(5)
        }
        assert 1 == len(outputs)

    def test_nested_dict_list_reports_exact_changes(self):
        """Reordered nested dicts pair by content: exactly one entry, no more, no less."""
        expected = {'values_changed': {
            "root['services'][1]['limits']['cpu']": {'new_value': '2', 'old_value': '1'},
        }}
        result = DeepDiff(self.NESTED_T1, self.NESTED_T2, ignore_order=True)
        assert expected == result

    def test_unhashable_items_reordered_yield_empty_diff(self):
        """Dicts containing unhashable values (sets, nested lists) match by content."""
        t1 = {"items": [
            {"name": "a", "tags": {"x", "y"}, "matrix": [[1, 2], [3, 4]]},
            {"name": "b", "tags": {"z"}, "matrix": [[5]]},
        ]}
        t2 = {"items": [
            {"name": "b", "tags": {"z"}, "matrix": [[5]]},
            {"name": "a", "tags": {"y", "x"}, "matrix": [[1, 2], [3, 4]]},
        ]}
        assert {} == DeepDiff(t1, t2, ignore_order=True)

    def test_unhashable_items_report_exact_paths(self):
        t1 = {"items": [{"name": "a", "v": 1}, {"name": "b", "v": 2}, {"name": "c", "v": 3}]}
        t2 = {"items": [{"name": "c", "v": 30}, {"name": "a", "v": 1}, {"name": "b", "v": 20}]}
        expected = {'values_changed': {
            "root['items'][2]['v']": {'new_value': 30, 'old_value': 3},
            "root['items'][1]['v']": {'new_value': 20, 'old_value': 2},
        }}
        result = DeepDiff(t1, t2, ignore_order=True)
        assert expected == result
        assert {"root['items'][2]['v']", "root['items'][1]['v']"} == result.affected_paths

    def test_exclude_paths_behaves_like_physical_removal(self):
        """An excluded path must not influence how list items are ordered or paired.

        Excluding root['services'][0]['ts'] must give the same result as
        deleting that key from both inputs before comparing.
        """
        t1 = {"services": [{"name": "web", "ts": 100}, {"name": "api", "ts": 200}]}
        t2 = {"services": [{"name": "api", "ts": 999}, {"name": "web", "ts": 888}]}
        result = DeepDiff(t1, t2, ignore_order=True,
                          exclude_paths=["root['services'][0]['ts']"])

        t1_stripped = {"services": [{"name": "web"}, {"name": "api", "ts": 200}]}
        t2_stripped = {"services": [{"name": "api", "ts": 999}, {"name": "web"}]}
        expected = DeepDiff(t1_stripped, t2_stripped, ignore_order=True)
        assert expected == result
        assert {'values_changed': {
            "root['services'][1]['ts']": {'new_value': 999, 'old_value': 200},
        }} == result

    @pytest.mark.parametrize('exclude_paths', [
        ["root['services'][0]['ts']", "root['services'][1]['ts']"],
        ["root['services'][*]['ts']"],
    ])
    def test_exclude_paths_noisy_field_is_fully_ignored(self, exclude_paths):
        """When the only differences are on excluded paths, the diff is empty."""
        t1 = {"services": [{"name": "web", "ts": 100}, {"name": "api", "ts": 200}]}
        t2 = {"services": [{"name": "api", "ts": 999}, {"name": "web", "ts": 888}]}
        assert {} == DeepDiff(t1, t2, ignore_order=True, exclude_paths=exclude_paths)

    def test_exclude_paths_with_real_change_reports_only_real_change(self):
        t1 = {"services": [
            {"name": "web", "ts": 100, "port": 80},
            {"name": "api", "ts": 200, "port": 8080},
        ]}
        t2 = {"services": [
            {"name": "api", "ts": 999, "port": 8080},
            {"name": "web", "ts": 888, "port": 81},
        ]}
        exclude_paths = ["root['services'][0]['ts']", "root['services'][1]['ts']"]
        expected = {'values_changed': {
            "root['services'][0]['port']": {'new_value': 81, 'old_value': 80},
        }}
        result = DeepDiff(t1, t2, ignore_order=True, exclude_paths=exclude_paths)
        assert expected == result

    def test_json_output_stable_across_hash_seeds(self):
        """Regression baselines stay valid: identical bytes in fresh processes."""
        outputs = _run_script_with_hash_seeds(_LIST_OF_DICTS_SCRIPT, seeds=(0, 1, 2, 42))
        assert 1 == len(set(outputs)), "to_json() output differs across hash seeds: %r" % (outputs,)
        parsed = json.loads(outputs[0])
        assert 3 == len(parsed["values_changed"])

    def test_set_results_stable_across_hash_seeds(self):
        outputs = _run_script_with_hash_seeds(_SET_SCRIPT, seeds=(0, 1, 2, 42))
        assert 1 == len(set(outputs)), "to_json() output differs across hash seeds: %r" % (outputs,)
        parsed = json.loads(outputs[0])
        assert ["root['tags']['alpha']"] == parsed["set_item_removed"]
        assert 2 == len(parsed["set_item_added"])
