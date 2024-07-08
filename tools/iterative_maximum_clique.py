import os
import sys

import numpy as np

class IterativeMaximunCalique(object):
    def __init__(self,eps,min_samples) -> None:
        self.threshold=eps
        self.min_samples=min_samples

    def find_maximum_clique(self,graph):
        n=len(graph)
        max_clique=[]
        def expand(current_calique,candidates):
            nonlocal max_clique
            if not candidates:
                if len(current_calique)>len(max_clique):
                    max_clique=current_calique[:]
                return 
            if len(current_calique)+len(candidates)<=len(max_clique):
                return 
            for v in candidates:
                if all(graph[v][u] for u in current_calique):
                    new_candidates=[u for u in candidates if graph[v][u]]
                    expand(current_calique+[v],new_candidates)
        expand([],list(range(n)))
        return max_clique
    
    def test_maximum_clique(self,):
        graph=[
            [0,1,1,0],
            [1,0,1,1],
            [1,1,0,1],
            [0,1,1,0]
        ]
        result=self.find_maximum_clique(graph)
        print(result) # [0,1,2]
        return result
    
    def update_graph(self,graph,clique):
        """
        将已经成为clique的节点全部变成false
        """
        n=len(graph)
        for i in range(n):
            for j in range(n):
                if (i in clique) or (j in clique):
                    graph[i][j]=False
        return graph

    def fit_predict(
            self,
            cost_matrix
        ):
        """
        使用 eps将 cost_matrix转换为 无向图
        对该无向图求解最大团
        删除最大团的节点
        继续求解最大团
        直到最大团的大小不足 self.min_samples
        """
        graph=cost_matrix<self.threshold
        cliques=[]
        while True:
            clique=self.find_maximum_clique(graph)
            if len(clique)<self.min_samples:
                break
            print(f"=> find a clique")
            graph=self.update_graph(graph,clique)
            cliques.append(clique)
        labels=[-1 for _ in range(len(graph))]
        for label,clique in enumerate(cliques):
            for index in clique:
                labels[index]=label
        return labels
            