# Based on: https://github.com/cattaneod/PointNetVlad-Pytorch
import numpy as np
import torch
import matplotlib.pyplot as plt

def best_pos_distance(query, pos_vecs):
    num_pos = pos_vecs.shape[0]
    query_copies = query.repeat(int(num_pos), 1)
    # ((pos_vecs - query_copies) ** 2).sum(2)
    diff = ((pos_vecs - query_copies)**2).sum(1)
    max_pos_dist = diff.max()
    return max_pos_dist


def triplet_loss(tuple_vecs, config):
    tuple_vecs = torch.split(
        tuple_vecs, [1, config.positives_per_query, config.negatives_per_query])
    q_vec, pos_vecs, neg_vecs = tuple_vecs[0], tuple_vecs[1], tuple_vecs[2]
    max_pos_dist = best_pos_distance(q_vec, pos_vecs)

    num_neg = neg_vecs.shape[0]
    query_copies = q_vec.repeat(int(num_neg), 1)
    max_pos_dist = max_pos_dist.view(1, -1)
    max_pos_dist = max_pos_dist.repeat(int(num_neg), 1)

    neg_dists = ((neg_vecs - query_copies) ** 2).sum(1)
    loss = config.loss_margin_1 + max_pos_dist - \
        neg_dists.reshape((int(num_neg), 1))
    loss = loss.clamp(min=0.0)
    if config.lazy_loss:
        triplet_loss = loss.max()
    else:
        triplet_loss = loss.sum()

    return triplet_loss


def quadruplet_loss(tuple_vecs, config):
    # Split the input vectors
    tuple_vecs = torch.split(tuple_vecs, [1, config.positives_per_query, config.negatives_per_query, 1])
    q_vec, pos_vecs, neg_vecs, other_neg = tuple_vecs[0], tuple_vecs[1], tuple_vecs[2], tuple_vecs[3]

    # Calculate the positive distance -> 두개의 pos vector 중에서 가장 큰 distance를 찾는다.
    max_pos_dist = best_pos_distance(q_vec, pos_vecs)
    
    # 거리가 가장 큰 pos vector의 거리 값(max_pos_dist)을 neg의 개수만큼 복사한다.
    num_neg = neg_vecs.shape[0] # neg의 개수
    max_pos_dist = max_pos_dist.view(1, -1) # max_pos_dist를 1x1로 만든다.
    max_pos_dist = max_pos_dist.repeat(int(num_neg), 1) # max_pos_dist를 neg의 개수만큼 복사한다.

    # Calculate negative distances -> neg_dists를 모든 neg vector에 대해서 구해준다.
    query_copies = q_vec.repeat(int(num_neg), 1) # 쿼리 vector를 neg의 개수만큼 복사한다.
    neg_dists = ((neg_vecs - query_copies) ** 2).sum(1) # neg와 query의 distance를 계산한다.
    
    # Compute the loss -> 마진 + max_pos_dist - neg_dists
    loss = config.loss_margin_1 + max_pos_dist - neg_dists.reshape((int(num_neg), 1))
    loss = loss.clamp(min=0.0)
    # print(f"loss after clamp: {loss}")  # Monitor the loss value
    
    if config.lazy_loss:
        triplet_loss = loss.max()
    else:
        triplet_loss = loss.sum()

    # Second part of the loss
    other_neg_copies = other_neg.repeat(int(num_neg), 1)
    other_neg_dists = ((neg_vecs - other_neg_copies) ** 2).sum(1)
    
    second_loss = config.loss_margin_2 + max_pos_dist - other_neg_dists.reshape((int(num_neg), 1))
    second_loss = second_loss.clamp(min=0.0)
    
    if config.lazy_loss:
        second_loss = second_loss.max()
    else:
        second_loss = second_loss.sum()

    # Calculate total loss
    total_loss = triplet_loss + second_loss
    
    return total_loss
