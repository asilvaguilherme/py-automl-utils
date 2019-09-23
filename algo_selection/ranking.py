import numpy as np


def to_rank(performances):
    
    matrix = np.array(performances)
    n_algo, n_measures = matrix.shape
    rankings = np.zeros((n_measures,n_algo))
    
    for i in range(n_measures):
#         print(matrix)
        measure = matrix[:,i]
        ranking_position = 1
        while ranking_position <= n_algo:
            next_algo_index = np.argmax(measure)
            rankings[i,next_algo_index] = ranking_position
            measure[next_algo_index] = -np.inf
            ranking_position += 1
    
    return rankings   

def average_rank(rankings):
    
    avgs = []
    
    _, n_algo = rankings.shape
    
    for algo in range(n_algo):
        avgs.append(np.mean(rankings[:,algo]))
    
    one_ranking = [0]*n_algo
    ranking_position = 1
    while ranking_position <= n_algo:
        next_algo_index = np.argmin(avgs)
        one_ranking[next_algo_index] = ranking_position
        avgs[next_algo_index] = np.inf
        ranking_position += 1
        
    return one_ranking

def from_perform_to_rank(performance):
    
    n_measures = len(performance)
    copy_performance = performance.copy()
    
    one_ranking = [0]*n_measures
    ranking_position = 1
    while ranking_position <= n_measures:
        next_algo_index = np.argmin(copy_performance)
        one_ranking[next_algo_index] = ranking_position
        copy_performance[next_algo_index] = np.inf
        ranking_position += 1
        
    return one_ranking

def from_perform_to_rank2(performance):
    
    copy_performance = performance.copy()
    
    ranking = [0] * len(performance)
    
    for i in range(len(performance)):
        ranking[i] = np.argmax(copy_performance) + 1
        copy_performance[np.argmax(copy_performance)] = -np.inf
            
    return ranking


