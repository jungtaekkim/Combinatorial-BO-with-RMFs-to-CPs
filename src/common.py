import numpy as np


def convert_str_to_arr(target):
    assert isinstance(target, str)

    list_ = []
    for char in target:
        list_.append(float(char))
    return np.array(list_)

def convert_arr_to_str(target):
    assert isinstance(target, np.ndarray)
    str_ = ''

    for elem in target:
        str_ += str(int(elem))
    return str_

def convert_int_to_str(target, dim):
    return '{1:0{0}d}'.format(dim, int(bin(target)[2:]))

def convert_str_to_int(target):
    return int(target, 2)

def convert_arr_to_arr_sign(target):
    assert isinstance(target, np.ndarray)
    return 2.0 * target - 1

def check_zero_one(target):
    assert isinstance(target, str)

    for char in target:
        if not char == '0' and not char == '1':
            return False
    return True

def embedding(arr, dim_target, w):
    return np.dot(arr, w) / arr.shape[0]

def get_all_embeddings(dim_problem, dim_target, w):
    list_embeddings = []

    for elem in range(0, 2**dim_problem):
        arr_cur = convert_str_to_arr(convert_int_to_str(elem, dim_problem))
        embedding_ = embedding(arr_cur, dim_target, w)
        list_embeddings.append(embedding_)

    return list_embeddings
