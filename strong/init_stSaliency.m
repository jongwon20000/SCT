function [rf, result] = init_stSaliency(feature, mask)

N_TREE = 1;

[rf, result] = init_pgrf(feature, mask, N_TREE,[]);
