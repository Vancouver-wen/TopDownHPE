import os
import sys

import numpy as np 

class DisjointSetCluster(object):
    def __init__(
            self,
            eps,
            min_samples,
        ):
        self.eps=eps
        self.min_samples=min_samples
    def fit_predict(
            self,
            cost_matrix
        ):
        """
        初始化一个空list
        先对所有的边进行排序(cost从小到大)
        对一条边 [n_i,n_j]
        1. n_i 与 n_j 不在list中
           list.append([n_i,n_j])
        2. n_i 与 n_j 其中一个在list中,假设n_i在list中,n_j不在list中
           if n_j与n_i所在list中的每个点距离都小于threshold=self.eps:
               把n_j添加到n_i所在的list中
            else:
                if n_j与其他某些集合满足约束:
                    将n_j加入到其他集合中
                else:
                    n_j单独成为一个list
        3. 如果 n_i 与 n_j 都在list中
           打印 边的cost 并 舍弃该边
        """
        height,width=cost_matrix.shape
        assert height==width,f"cost matrix should satisfy height==width"
        node_num=height
        edges=[]
        for i in range(0,node_num):
            for j in range(i+1,node_num):
                edges.append([i,j,cost_matrix[i][j]])
        edges=sorted(edges,key=lambda x:x[-1])
        edges=np.array(edges)[:,:-1].astype(np.int32)
        disjoint_sets=[]
        for edge in edges:
            n_i,n_j=edge
            i_in=self.node_in_disjoint_sets(n_i,disjoint_sets)
            j_in=self.node_in_disjoint_sets(n_j,disjoint_sets)
            if (i_in or j_in)==False:
                disjoint_sets.append([n_i,n_j])
            else:
                # 此时 i_in or j_in 为 True
                if (i_in and j_in)==False:
                    # i_in 与 j_in 一个True一个False
                    if i_in==False: # 假设 n_i 在 disjoint sets 中
                        n_i,n_j=n_j,n_i
                        i_in,j_in=j_in,i_in
                    disjoint_set=self.find_disjoint_set_with_node(n_i,disjoint_sets)
                    is_in=self.is_node_in_disjoint_set(n_j,disjoint_set,cost_matrix)
                    if is_in:
                        disjoint_set.append(n_j)
                    else:
                        has_in=False
                        for disjoint_set in disjoint_sets:
                            if self.is_node_in_disjoint_set(n_j,disjoint_set,cost_matrix):
                                disjoint_set.append(n_j)
                                has_in=True
                                break
                        if not has_in:
                            disjoint_sets.append([n_j])
                else:
                    # 此时 i_in and j_in 为 True
                    print(f"encount conflict! drop edge<{n_i},{n_j}>")
        labels=self.convert_disjoint_sets_to_labels(disjoint_sets,node_num)
        return labels
    
    def node_in_disjoint_sets(self,node,disjoint_sets):
        for disjoint_set in disjoint_sets:
            if node in disjoint_set:
                return True
        return False
    
    def find_disjoint_set_with_node(self,node,disjoint_sets):
        for disjoint_set in disjoint_sets:
            if node in disjoint_set:
                return disjoint_set
        assert False,f"can not find node in disjoint sets"
    
    def is_node_in_disjoint_set(self,node,disjoint_set,cost_matrix):
        for disjoint in disjoint_set:
            cost=cost_matrix[node][disjoint]
            if cost>self.eps:
                return False
        return True

    def convert_disjoint_sets_to_labels(self,disjoint_sets,node_num):
        labels=np.zeros(node_num)-1
        cls_num=0
        for disjoint_set in disjoint_sets:
            if len(disjoint_set)<self.min_samples:
                continue
            for disjoint in disjoint_set:
                labels[disjoint]=cls_num
            cls_num+=1
        return labels


if __name__=="__main__":
    pass