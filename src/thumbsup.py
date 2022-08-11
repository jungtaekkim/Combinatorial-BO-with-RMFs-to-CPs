import numpy as np

import common


def objective(target):
    print(target)
    assert common.check_zero_one(target)
    bx = common.convert_str_to_arr(target)
    val = -1.0 * np.sum(bx)
    val /= bx.shape[0]
    return val


if __name__ == '__main__':
    output = objective('101001')
    print(output)
