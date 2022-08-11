import numpy as np
import copy
from bayeso import bo

import common
import seesaw


def get_fun(dim, seed):
    weights = seesaw.get_weights(dim, seed)

    fun_objective = lambda target: seesaw.objective(target, weights)

    return fun_objective

def _bo(model_bo, fun_objective, C_init, X_init, Y_init, num_iter, dim, embeddings_all):
    # X_init, X_init, Y_init: 1d lists
    C_final = copy.deepcopy(C_init)
    X_final = copy.deepcopy(X_init)
    Y_final = copy.deepcopy(Y_init)

    for ind_iter in range(0, num_iter):
        print('{} iteration'.format(ind_iter + 1))
        next_point, _ = model_bo.optimize(
            np.array(X_final),
            np.expand_dims(np.array(Y_final), axis=1)
        )
        print(next_point)

        norms_ = np.linalg.norm(embeddings_all - next_point, axis=1, ord=2)
        next_point = np.argmin(norms_)

        while common.convert_int_to_str(next_point, dim) in C_final:
            norms_[next_point] = np.inf
            next_point = np.argmin(norms_)

        embedding_selected = embeddings_all[next_point]
        combination_selected = common.convert_int_to_str(next_point, dim)
        print(combination_selected)

        C_final.append(combination_selected)
        X_final.append(embedding_selected)
        Y_final.append(fun_objective(combination_selected))

    return C_final, X_final, Y_final

def bo_ours(dim_problem, seed, num_iter, dim_target, num_init):
    fun_objective = get_fun(dim_problem, seed)

    seed_embedding = (seed + 1) * 19 + 1001
    w = np.random.RandomState(seed_embedding).rand(dim_problem, dim_target)
    print(w)
    embeddings_all = np.array(common.get_all_embeddings(dim_problem, dim_target, w))

    np.random.seed((seed + 1) * 13 + 101)
    choices = np.random.choice(2**dim_problem, num_iter, replace=False)
    choice = '{1:0{0}d}'.format(dim_problem, int(bin(choices[0])[2:]))

    minimizers = [choice]
    minimizers_arr = [embeddings_all[common.convert_str_to_int(choice)]]
    minima = [fun_objective(choice)]

    model_bo = bo.BO(np.array([[0.0, 1.0]] * dim_target), str_acq='ucb', str_cov='matern52', normalize_Y=False)

    minimizers, minimizers_arr, minima = _bo(model_bo, fun_objective, minimizers, minimizers_arr, minima, num_iter - num_init, dim_problem, embeddings_all)

    return minimizers, minima


if __name__ == '__main__':
    list_all = []

    num_bo = 10
    num_init = 1

    dim_problem = 16
    dim_target = 20

    for seed in range(0, num_bo):
        print(dim_problem, dim_target, seed)
        minimizers, minima = bo_ours(dim_problem, seed, np.minimum(100 + num_init, 2**dim_problem), dim_target, num_init)
        print(minima)

        dict_exp = {
            'dim_problem': dim_problem,
            'dim_target': dim_target,
            'seed': seed,
            'minimizers': minimizers,
            'minima': minima,
        }
        list_all.append(dict_exp)

    np.save('bo_ours_seesaw.npy', list_all)
