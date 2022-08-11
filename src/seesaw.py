import numpy as np

import common


def get_weights(dim, seed):
    return np.random.RandomState((seed + 1) * 42 + 111).uniform(low=0.5, high=2.5, size=dim)

def objective(target, weights):
    assert common.check_zero_one(target)
    bx = common.convert_str_to_arr(target)

    len_bx = bx.shape[0]
    len_seesaw = len_bx / 2
    assert bx.shape[0] == weights.shape[0]

    if (len_bx // 2) != len_seesaw:
        raise ValueError

    val = 0.0

    if np.sum(bx) == 0:
        return 1e5

    for ind_elem, elem in enumerate(bx):
        if ind_elem >= len_seesaw:
            ind_ = ind_elem + 1
        else:
            ind_ = ind_elem
            
        val += (ind_ - len_seesaw) * elem * weights[ind_elem]

    return np.abs(val)


if __name__ == '__main__':
    weights = get_weights(6, 1)
    print(weights)

    output = objective('101001', weights)
    print(output)

    weights = get_weights(6, 42)
    print(weights)

    output = objective('101001', weights)
    print(output)
