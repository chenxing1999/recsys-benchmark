import math

from src.metrics import get_ndcg, get_ndcg_recall


def test_ndcg_simple():
    true_items = [[2, 4, 5, 10]]
    pred_items = [[1, 2, 3, 4, 5, 6, 7]]

    ndcg = get_ndcg(pred_items, true_items, k=7)

    idcg = sum([1 / math.log2(x + 2) for x in range(4)])

    expected_dcg = 1 / math.log2(3) + 1 / math.log2(5) + 1 / math.log2(6)
    expected = expected_dcg / idcg

    assert math.isclose(ndcg, expected)


def test_ndcg_more_pred_item():
    true_items = [[2, 4, 5, 10]]
    pred_items = [[1, 2, 3, 4, 5, 6, 7]]

    ndcg = get_ndcg(pred_items, true_items, k=4)

    idcg = sum([1 / math.log2(x + 2) for x in range(4)])

    expected_dcg = 1 / math.log2(3) + 1 / math.log2(5)
    expected = expected_dcg / idcg

    assert math.isclose(ndcg, expected)


def test_ndcg_true_item_set():
    true_items = [set([2, 4, 5, 10])]
    pred_items = [[1, 2, 3, 4, 5, 6, 7]]

    ndcg = get_ndcg(pred_items, true_items, k=4)

    idcg = sum([1 / math.log2(x + 2) for x in range(4)])

    expected_dcg = 1 / math.log2(3) + 1 / math.log2(5)
    expected = expected_dcg / idcg

    assert math.isclose(ndcg, expected)


def test_ndcg_multiple_users():
    true_items = [
        set([2, 4, 5, 10]),
        set([7, 8, 9]),
    ]
    pred_items = [[1, 2, 3, 4, 5, 6, 7], [8, 2, 3, 4, 5, 6, 7]]

    ndcg = get_ndcg(pred_items, true_items, k=4)

    # Calculate expected for 1
    idcg1 = sum([1 / math.log2(x + 2) for x in range(4)])

    expected_dcg1 = 1 / math.log2(3) + 1 / math.log2(5)
    expected1 = expected_dcg1 / idcg1

    # calculate expected for 2
    idcg2 = sum([1 / math.log2(x + 2) for x in range(3)])

    expected_dcg2 = 1 / math.log2(2)
    expected2 = expected_dcg2 / idcg2

    expected = (expected1 + expected2) / 2

    assert math.isclose(ndcg, expected)


def test_ndcg_recall_multiple_users():
    true_items = [
        set([2, 4, 5, 10]),
        set([7, 8, 9]),
    ]
    pred_items = [[1, 2, 3, 4, 5, 6, 7], [8, 2, 3, 4, 5, 6, 7]]

    ndcg, recall = get_ndcg_recall(pred_items, true_items, k=4)

    # Calculate expected for 1
    idcg1 = sum([1 / math.log2(x + 2) for x in range(4)])

    expected_dcg1 = 1 / math.log2(3) + 1 / math.log2(5)
    expected1 = expected_dcg1 / idcg1

    # calculate expected for 2
    idcg2 = sum([1 / math.log2(x + 2) for x in range(3)])

    expected_dcg2 = 1 / math.log2(2)
    expected2 = expected_dcg2 / idcg2

    expected = (expected1 + expected2) / 2

    assert math.isclose(ndcg, expected)

    # Calculate expected recall
    expected = (2 / 4 + 1 / 3) / 2
    assert math.isclose(recall, expected)


def test_recall_copy():
    """Copy test case from
    github.com/MaurizioFD/RecSys2019_DeepLearning_Evaluation
    """

    pos_items = [[2, 4, 5, 10]]
    ranked_list_1 = [[1, 2, 3, 4, 5]]
    ranked_list_2 = [[10, 5, 2, 4, 3]]
    ranked_list_3 = [[1, 3, 6, 7, 8]]

    recall = get_ndcg_recall(ranked_list_1, pos_items)[1]
    assert math.isclose(recall, 3 / 4)

    recall = get_ndcg_recall(ranked_list_2, pos_items)[1]
    assert math.isclose(recall, 1.0)

    recall = get_ndcg_recall(ranked_list_3, pos_items)[1]
    assert math.isclose(recall, 0.0)
