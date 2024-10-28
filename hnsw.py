#!python3
import sys
import numpy as np
import time
import random
from math import log2, log10
from heapq import heapify, heappop, heappush, heapreplace, nlargest, nsmallest

def l2_distance(a, b):
    return np.linalg.norm(a - b)

def heuristic(candidates, curr, k, distance_func, data):
    candidates = sorted(candidates, key=lambda a: a[1])
    result_indx_set = {candidates[0][0]}
    result = [candidates[0]]
    added_data = [ data[candidates[0][0]] ]
    for c, curr_dist in candidates[1:]:
        c_data = data[c]       
        if curr_dist < min(map(lambda a: distance_func(c_data, a), added_data)):
            result.append( (c, curr_dist))
            result_indx_set.add(c)
            added_data.append(c_data)
    for c, curr_dist in candidates: # optional. uncomment to build neighborhood exactly with k elements.
        if len(result) < k and (c not in result_indx_set):
            result.append( (c, curr_dist) )
    
    return result
def k_closest(candidates: list, curr, k, distance_func, data):
    return sorted(candidates, key=lambda a: a[1])[:k]
    
class HNSW:
    # self._graphs[level][i] contains a {j: dist} dictionary,
    # where j is a neighbor of i and dist is distance

    def _distance(self, x, y):
        return self.distance_func(x, [y])[0]

    def vectorized_distance_(self, x, ys):
        return [self.distance_func(x, y) for y in ys]

    def __init__(self, distance_func, m=5, ef=10, ef_construction=30, m0=None, neighborhood_construction=heuristic, vectorized=False):
        self.data = []
        self.distance_func = distance_func
        self.neighborhood_construction = neighborhood_construction

        if vectorized:
            self.distance = self._distance
            self.vectorized_distance = distance_func
        else:
            self.distance = distance_func
            self.vectorized_distance = self.vectorized_distance_

        self._m = m
        self._ef = ef
        self._ef_construction = ef_construction
        self._m0 = 2 * m if m0 is None else m0
        self._level_mult = 1 / log10(m)
        self._graphs = []
        self._enter_point = None

        self._median_points = []

    # def calculate_median_points(self):
    #     if len(self.data) == 0:
    #         return
        
    #     sample_data = np.array(random.sample(self.data, min(len(self.data), 1000)))
    #     self._median_points = np.median(sample_data, axis=0).reshape(-1, len(sample_data[0]))

    # def find_closest_median(self, elem):
    #     distances = [self.distance_func(elem, median_point) for median_point in self._median_points]
    #     return min(distances)

    def add(self, elem, ef=None):

        if ef is None:
            ef = self._ef

        distance = self.distance
        data = self.data
        graphs = self._graphs
        point = self._enter_point
        # print("enter point: ", self._enter_point)
        m = self._m

        # level at which the element will be inserted
        # closest_median_dist = self.find_closest_median(elem)
        # if closest_median_dist < np.median([self.find_closest_median(p) for p in data]):
            # level = int(-log2(random.random()) * self._level_mult) + 2
        # else:
        level = int(-log2(random.random()) * self._level_mult) + 1
        # print("level: %d" % level)

        # elem will be at data[idx]
        idx = len(data)
        data.append(elem)


        if point is not None:  # the HNSW is not empty, we have an entry point
            dist = distance(elem, data[point])
            # for all levels in which we dont have to insert elem,
            # we search for the closest neighbor
            for layer in reversed(graphs[level:]):
                point, dist = self.beam_search(graph=layer, q=elem, k=1, eps=[point], ef=1)[0]
            # at these levels we have to insert elem; ep is a heap of entry points.

            layer0 = graphs[0]
            for layer in reversed(graphs[:level]):
                level_m = m if layer is not layer0 else self._m0
                # navigate the graph and update ep with the closest
                # nodes we find
                # ep = self._search_graph(elem, ep, layer, ef)
                candidates = self.beam_search(graph=layer, q=elem, k=level_m*2, eps=[point], ef=self._ef_construction)
                point = candidates[0][0]
                
                # insert in g[idx] the best neighbors
                # layer[idx] = layer_idx = {}
                # self._select(layer_idx, ep, level_m, layer, heap=True)

                neighbors = self.neighborhood_construction(candidates=candidates, curr=idx, k=level_m, distance_func=self.distance_func, data=self.data)
                layer[idx] = neighbors
                # insert backlinks to the new node
                for j, dist in neighbors:
                    candidates_j = layer[j] + [(idx, dist)]
                    neighbors_j = self.neighborhood_construction(candidates=candidates_j, curr=j, k=level_m, distance_func=self.distance_func, data=self.data)
                    layer[j] = neighbors_j
                    
                
        for i in range(len(graphs), level):
            # for all new levels, we create an empty graph
            graphs.append({idx: []})
            self._enter_point = idx

    def calculate_median_points(self, data, num_medians=1):
        median_points = np.median(data, axis=0)

        if num_medians > 1:
            medians = []
            for _ in range(num_medians):
                random_point = median_points + np.random.normal(0, 0.01, size=median_points.shape)
                medians.append(random_point)
            return np.array(medians)
        return np.array([median_points])
        # return median_points
        
    def find_closest_indices(self, data, points):
        closest_indices = []
        for point in points:
            distances = np.linalg.norm(data - point, axis=1)
            closest_index = int(np.argmin(distances))
            closest_indices.append(closest_index)
        return closest_indices
    
    # can be used for search after jump        
    def search(self, q, k=1, ef=10, level=0, return_observed=True):
        graphs = self._graphs
        point = self._enter_point
        median_points = self.calculate_median_points(self.data, num_medians=1)
        median_points_indices = self.find_closest_indices(self.data, median_points)
        median_points = median_points_indices
        
        # print("start point", [point])
        for layer in reversed(graphs[level:]):
            # print(len(graphs))
            # print(len([l for l in layer]))
            # print("point:", point)
            point, dist = self.beam_search(layer, q=q, k=1, eps=[point], ef=1)[0]


        return self.beam_search(graph=graphs[level], q=q, k=k, eps=[point], ef=ef, return_observed=return_observed)

    def beam_search(self, graph, q, k, eps, ef, ax=None, marker_size=20, return_observed=False):
        '''
        graph – the layer where the search is performed
        q - query
        k - number of closest neighbors to return
        eps – entry points [vertex_id, ..., vertex_id]
        ef – size of the beam
        observed – if True returns the full of elements for which the distance were calculated
        returns – a list of tuples [(vertex_id, distance), ... , ]
        '''
        # Priority queue: (negative distance, vertex_id)
        candidates = []
        visited = set()  # set of vertex used for extending the set of candidates
        observed = dict() # dict: vertex_id -> float – set of vertexes for which the distance were calculated
        # print(len(graph))
        if ax:
            ax.scatter(x=q[0], y=q[1], s=marker_size, color='red', marker='^')
            ax.annotate('query', (q[0], q[1]))

        # Initialize the queue with the entry points
        for ep in eps:
            dist = self.distance_func(q, self.data[ep])
            heappush(candidates, (dist, ep))
            observed[ep] = dist

        while candidates:
            # Get the closest vertex (furthest in the max-heap sense)
            dist, current_vertex = heappop(candidates)
            # print(current_vertex)
            if ax:
                ax.scatter(x=self.data[current_vertex][0], y=self.data[current_vertex][1], s=marker_size, color='red')
                ax.annotate( len(visited), self.data[current_vertex] )

            # check stop conditions #####
            observed_sorted = sorted( observed.items(), key=lambda a: a[1] )
            # print(observed_sorted)
            ef_largets = observed_sorted[ min(len(observed)-1, ef-1 ) ]
            # print(ef_largets[0], '<->', -dist)
            if ef_largets[1] < dist:
                break
            #############################

            # Add current_vertex to visited set
            visited.add(current_vertex)

            # Check the neighbors of the current vertex
            for neighbor, _ in graph[current_vertex]:
                if neighbor not in observed:
                    dist = self.distance_func(q, self.data[neighbor])                    
                    # if neighbor not in visited:
                    heappush(candidates, (dist, neighbor))
                    observed[neighbor] = dist                    
                    if ax:
                        ax.scatter(x=self.data[neighbor][0], y=self.data[neighbor][1], s=marker_size, color='yellow')
                        # ax.annotate(len(visited), (self.data[neighbor][0], self.data[neighbor][1]))
                        ax.annotate(len(visited), self.data[neighbor])
                    
        
        # Sort the results by distance and return top-k
        observed_sorted =sorted( observed.items(), key=lambda a: a[1] )
        if return_observed:
            return observed_sorted
        return observed_sorted[:k]
    def save_graph_plane(self, file_path):
        with open(file_path, "w") as f:
            f.write(f'{len(self.data)}\n')

            for x in self.data:
                s = ' '.join([a.astype('str') for a in x ])
                f.write(f'{s}\n')

            for graph in self._graphs:
                for src, neighborhood in graph.items():
                    for dst, dist in neighborhood: 
                        f.write(f'{src} {dst}\n')



# n = int(sys.argv[1]) # graph size
# dim = int(sys.argv[2]) # vector dimensionality
# m = int(sys.argv[3]) # avg number of vertex
# m0 = int(sys.argv[3]) # avg number of vertex for the lower layer

# hnsw = HNSW( distance_func=l2_distance, m=5, m0=7, ef=10, ef_construction=30,  neighborhood_construction = heuristic)

# k =5 
# dim = 2
# n = 1000
# data = np.array(np.float32(np.random.random((n, dim))))


# for x in data:
#     hnsw.add(x)
