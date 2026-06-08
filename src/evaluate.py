import faiss
import numpy as np
import math
from collections import defaultdict
import numba as nb
from numba import prange


@nb.njit()
def compute_ranking_metrics(testusers, testdata, traindata, topk_list, user_rank_pred_items):
    all_metrics = []
    for i in prange(len(testusers)):
        u = testusers[i]
        one_metrics = []
        mask_items = traindata[i]
        test_items = testdata[i]
        pos_length = len(test_items)
        pred_items_all = user_rank_pred_items[u]
        max_length_candicate = len(mask_items) + topk_list[-1]
        pred_items = [item for item in pred_items_all[:max_length_candicate] if item not in mask_items][:topk_list[-1]]
        for topk in topk_list:
            hit_value = 0
            dcg_value = 0
            for idx in prange(topk):
                if pred_items[idx] in test_items:
                    hit_value += 1
                    dcg_value += math.log(2) / math.log(idx + 2)
            target_length = min(topk, pos_length)
            idcg = 0.0
            for k in prange(target_length):
                idcg = idcg + math.log(2) / math.log(k + 2)
            hr_cur = 1.0 if hit_value > 0 else 0.0
            recall_cur = hit_value / pos_length
            ndcg_cur = dcg_value / idcg
            one_metrics.append([hr_cur, recall_cur, ndcg_cur])
        all_metrics.append(one_metrics)
    return all_metrics


def num_faiss_evaluate(_test_ratings, _train_ratings, _topk_list, _user_matrix, _item_matrix, _test_users):
    hr_topk_list = defaultdict(list)
    recall_topk_list = defaultdict(list)
    ndcg_topk_list = defaultdict(list)
    hr_out, recall_out, ndcg_out = {}, {}, {}

    test_users = _test_users
    dim = _user_matrix.shape[-1]
    index = faiss.IndexFlatIP(dim)
    index.add(_item_matrix)
    max_mask_items_length = max(len(_train_ratings[user]) for user in _train_ratings.keys())
    sim, _user_rank_pred_items = index.search(_user_matrix, _topk_list[-1] + max_mask_items_length)

    testdata = [list(_test_ratings[user]) for user in test_users]
    traindata = [list(_train_ratings[user]) if user in _train_ratings.keys() else [-1] for user in test_users]
    all_metrics = compute_ranking_metrics(nb.typed.List(test_users), nb.typed.List(testdata),
                                          nb.typed.List(traindata), nb.typed.List(_topk_list),
                                          nb.typed.List(_user_rank_pred_items))

    for i, one_metrics in enumerate(all_metrics):
        j = 0
        for topk in _topk_list:
            hr_topk_list[topk].append(one_metrics[j][0])
            recall_topk_list[topk].append(one_metrics[j][1])
            ndcg_topk_list[topk].append(one_metrics[j][2])
            j += 1
    for topk in _topk_list:
        recall_out[topk] = np.mean(recall_topk_list[topk])
        hr_out[topk] = np.mean(hr_topk_list[topk])
        ndcg_out[topk] = np.mean(ndcg_topk_list[topk])
    return hr_out, recall_out, ndcg_out


def build_5group_segments(item_list_sorted):
    """Split items into 5 equal groups. item_list_sorted: ascending by interaction count."""
    n = len(item_list_sorted)
    seg = max(1, n // 5)
    return {
        'tail':     list(item_list_sorted[:seg]),
        'knee':     list(item_list_sorted[seg:2*seg]),
        'body':     list(item_list_sorted[2*seg:3*seg]),
        'shoulder': list(item_list_sorted[3*seg:4*seg]),
        'head':     list(item_list_sorted[4*seg:]),
    }


def _segment_only_faiss_evaluate(_test_ratings, _train_ratings, _topk_list,
                                  _user_matrix, _item_matrix, _test_users, _segment_items):
    hr_topk_list = defaultdict(list)
    recall_topk_list = defaultdict(list)
    ndcg_topk_list = defaultdict(list)
    hr_out, recall_out, ndcg_out = {}, {}, {}

    segment_items = [int(item) for item in _segment_items]
    if len(segment_items) == 0:
        for topk in _topk_list:
            hr_out[topk] = 0.0
            recall_out[topk] = 0.0
            ndcg_out[topk] = 0.0
        return hr_out, recall_out, ndcg_out

    orig_to_local = {item: idx for idx, item in enumerate(segment_items)}
    test_users = list(_test_users)
    local_testdata = []
    local_traindata = []
    for user in test_users:
        local_testdata.append({
            orig_to_local[int(item)]
            for item in _test_ratings.get(user, [])
            if int(item) in orig_to_local
        })
        local_traindata.append({
            orig_to_local[int(item)]
            for item in _train_ratings.get(user, [])
            if int(item) in orig_to_local
        })

    max_mask_items_length = max((len(items) for items in local_traindata), default=0)
    search_k = min(len(segment_items), max(_topk_list) + max_mask_items_length)

    query_vectors = np.ascontiguousarray(_user_matrix.astype('float32', copy=False))
    segment_matrix = np.ascontiguousarray(
        _item_matrix[np.asarray(segment_items, dtype=np.int64)].astype('float32', copy=False)
    )
    dim = segment_matrix.shape[-1]
    index = faiss.IndexFlatIP(dim)
    index.add(segment_matrix)
    _, user_rank_pred_items = index.search(query_vectors, search_k)

    for row_idx, user in enumerate(test_users):
        mask_items = local_traindata[row_idx]
        test_items = local_testdata[row_idx]
        pred_items = [
            int(item)
            for item in user_rank_pred_items[user]
            if int(item) >= 0 and int(item) not in mask_items
        ]
        pos_length = len(test_items)
        for topk in _topk_list:
            hit_value = 0
            dcg_value = 0.0
            for idx, item in enumerate(pred_items[:topk]):
                if item in test_items:
                    hit_value += 1
                    dcg_value += math.log(2) / math.log(idx + 2)

            if pos_length == 0:
                hr_cur, recall_cur, ndcg_cur = 0.0, 0.0, 0.0
            else:
                target_length = min(topk, pos_length)
                idcg = sum(math.log(2) / math.log(k + 2) for k in range(target_length))
                hr_cur = 1.0 if hit_value > 0 else 0.0
                recall_cur = hit_value / target_length
                ndcg_cur = dcg_value / idcg

            hr_topk_list[topk].append(hr_cur)
            recall_topk_list[topk].append(recall_cur)
            ndcg_topk_list[topk].append(ndcg_cur)

    for topk in _topk_list:
        recall_out[topk] = float(np.mean(recall_topk_list[topk]))
        hr_out[topk] = float(np.mean(hr_topk_list[topk]))
        ndcg_out[topk] = float(np.mean(ndcg_topk_list[topk]))
    return hr_out, recall_out, ndcg_out


def num_faiss_evaluate_head_tail(_test_ratings, _train_ratings, _topk_list,
                                  _user_matrix, _item_matrix, _test_users, _head_items, _tail_items):
    hr_out_h, recall_out_h, ndcg_out_h = _segment_only_faiss_evaluate(
        _test_ratings, _train_ratings, _topk_list,
        _user_matrix, _item_matrix, _test_users, _head_items)
    hr_out_t, recall_out_t, ndcg_out_t = _segment_only_faiss_evaluate(
        _test_ratings, _train_ratings, _topk_list,
        _user_matrix, _item_matrix, _test_users, _tail_items)
    return hr_out_h, recall_out_h, ndcg_out_h, hr_out_t, recall_out_t, ndcg_out_t


def num_faiss_evaluate_hsbt(_test_ratings, _train_ratings, _topk_list,
                             _user_matrix, _item_matrix, _test_users,
                             _head_items, _shoulder_items, _body_items, _knee_items, _tail_items):
    """5-group evaluation. Returns {segment: {K: (hr, ndcg)}}."""
    segment_items = {
        'head': _head_items,
        'shoulder': _shoulder_items,
        'body': _body_items,
        'knee': _knee_items,
        'tail': _tail_items,
    }
    result = {}
    for seg, items in segment_items.items():
        hr_out, _, ndcg_out = _segment_only_faiss_evaluate(
            _test_ratings, _train_ratings, _topk_list,
            _user_matrix, _item_matrix, _test_users, items)
        result[seg] = {topk: (hr_out[topk], ndcg_out[topk]) for topk in _topk_list}
    return result
