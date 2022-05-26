import numpy as np
import torch
import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
import cpp_wrappers.cpp_neighbors.radius_neighbors as cpp_neighbors

def batch_grid_subsampling_kpconv(points,\
                                  batches_len,\
                                  sampleDl=0.1,\
                                  max_p=0):
    s_points, s_len = cpp_subsampling.subsample_batch(points,
                                                      batches_len,
                                                      sampleDl=sampleDl,
                                                      max_p=max_p,
                                                      verbose=0)
    return s_points, s_len


def batch_neighbors_kpconv(queries, supports, q_batches, s_batches, radius, max_neighbors):
    """
    Computes neighbors for a batch of queries and supports
    :param queries: (N1, 3) the query points
    :param supports: (N2, 3) the support points
    :param q_batches: (B) the list of lengths of batch elements in queries
    :param s_batches: (B)the list of lengths of batch elements in supports
    :param radius: float32
    :return: neighbors indices
    """

    neighbors = cpp_neighbors.batch_query(queries, supports, q_batches, s_batches, radius=radius)
    if max_neighbors > 0:
        return neighbors[:, :max_neighbors]
    else:
        return neighbors

def prepare_input(points,N,neighborhood_limits,r_init=0.01):
    #points: [N_k-3]_K; K pointclouds each with N_k points.
    #N:int num layers
    neighborhood_limits=neighborhood_limits
    
    batched_points = np.concatenate(points, axis=0).astype('float32')
    batched_lengths = np.concatenate([np.array([x.shape[0]]) for x in points], axis=0).astype('int32')
    #batched_points: N_full-3
    #batched_lengths: K
    r = r_init*2.5#init grid size = 0.01, 2.5 kernel size
    #prepare input
    input_points = []
    input_neighbors = []
    input_pools = []
    input_upsamples = []
    input_batches_len = []
    #start
    for i in range(N):
        conv_i = batch_neighbors_kpconv(batched_points, batched_points, batched_lengths, batched_lengths, r, neighborhood_limits[i])
        dl = 2*r/2.5
        sub_points, sub_length = batch_grid_subsampling_kpconv(batched_points, batched_lengths, sampleDl=dl)
        pool_i = batch_neighbors_kpconv(sub_points, batched_points, sub_length, batched_lengths, r, neighborhood_limits[i])
        up_i = batch_neighbors_kpconv(batched_points, sub_points, batched_lengths, sub_length, 2 * r, neighborhood_limits[i])
        #collect
        input_points += [batched_points.astype('float32')]
        input_neighbors += [conv_i.astype('int32')]
        input_pools += [pool_i.astype('int32')]
        input_upsamples += [up_i.astype('int32')]
        input_batches_len += [batched_lengths.astype('int32')]
        # New points for next layer
        batched_points = sub_points.astype('float32')
        batched_lengths = sub_length.astype('int32')
        r *= 2
    dict_inputs = {
        'points': input_points,
        'neighbors': input_neighbors,
        'pools': input_pools,
        'upsamples': input_upsamples,
        'stack_lengths': input_batches_len
    }
    return dict_inputs

def np2torch(batched_input,features_list,device):
    batched_input_torch = dict()
    points_list = [torch.Tensor(x).float().to(device) for x in batched_input['points']]
    batched_input_torch['points'] = points_list
    neighbors_list = [torch.Tensor(x).long().to(device) for x in batched_input['neighbors']]
    batched_input_torch['neighbors'] = neighbors_list
    pools_list = [torch.Tensor(x).long().to(device) for x in batched_input['pools']]
    batched_input_torch['pools'] = pools_list
    upsamples_list = [torch.Tensor(x).long().to(device) for x in batched_input['upsamples']]
    batched_input_torch['upsamples'] = upsamples_list
    stack_lengths_list = [torch.Tensor(x).int().to(device) for x in batched_input['stack_lengths']]
    batched_input_torch['stack_lengths'] = stack_lengths_list
    batched_input_torch['features'] = features_list
    return batched_input_torch
