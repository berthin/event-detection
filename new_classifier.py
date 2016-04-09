import numpy as np
from sklearn.neighbors import DistanceMetric
from sklearn.externals.joblib import delayed, Parallel
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import chi2_kernel
import sys

class unname_classifier:
  """
  Initalize the class. Receives data_train and its labels. Distance (by the moment) will only be used for predicting the labels, but for creating the clusters by default kmeans works with euclidean distance
  :::::::::
  data  : n_classes x n_points_per_class x n_features_per_point
  label : n_classes x 1
  """
  def __init__ (self, data, label, distance = 'euclidean', params_distance = None):
    class special_distance:
      # For mahalanobis, seuclidean
      def __init__ (self, distance):
        self.distance = distance
      def pairwise (self, (x, y)):
        if self.distance[0] == 'm':
          V = np.cov (x, y)
          return DistanceMetric.get_metric (self.distance, V=V).pairwise (V)
        elif self.distance[0] == 's':
          V = np.var ((x, y), axis = 0)
          return DistanceMetric.get_metric (self.distance, V=V).pairwise ((x, y))
        elif self.distance[0] == 'c':
          return chi2_kernel ((x, y))
        else:
          return np.array ([[0, -np.dot (x,y) / (np.dot (x,x) + np.dot (y,y) - np.dot (x,y))]])

    self.data_init = data
    self.label = label

    if distance in ['euclidean', 'manhattan', 'hamming', 'canberra', 'braycurtis']:
      self.f_distance = DistanceMetric.get_metric (distance)
    elif distance in ['chebyshev', 'minkowski', 'wminkowski']:
      self.f_distance = DistanceMetric.get_metric (distance, params_distance)
    elif distance in ['seuclidean', 'mahalanobis', 'chi2', 'tanimoto']:
      self.f_distance = special_distance (distance)
    else:
      print 'Error, no distance metric has been set up'

  """
  Find the best k-clusters for the data. This method estimates k
  ::::
  data  : n_points_per_class x n_features_per_point
  """
  def find_best_clusters (data, min_k = 2, max_k = -1, thr = 1, n_runs_kmeans = 5):
    max_k = np.min (len (data), 15)
    data_kmeans = np.array ([KMeans (init = 'k-means++', n_clusters = k, n_init = n_runs_kmeans).fit (data) for k in xrange (min_k - 1, max_k + 1)])
    inertia_kmeans = [km.inertia_ for km in data_kmeans]

    for i in xrange (1, len (inertia_kmeans)):
      print i, inertia_kmeans[i]
      if (inertia_kmeans [i - 1] - inertia_kmeans[i] > thr * len (data)):
        return data_kmeans[i].cluster_centers_
    print 'Best clusters not found...'
    return None

  """
  Fit the classifier finding the clusters for each class
  :::::
  data  : n_classes x n_points_per_class x n_features_per_point
  """
  def fit (self, n_processors = 1):
    self.data_clusters = np.array (Parallel (n_jobs = n_processors) (delayed (find_best_clusters) (data_class) for data_class in self.data_init))
    #print self.data_clusters.shape
    return self.data_clusters

  """
  Calculate the cost of assigning the point to a specific clusters' class
  :::::::::
  clusters  : n_clusters_per_class x n_features
  points    : n_single_query_points x n_features
  """
  def calculate_cost (self, clusters, points, C=1.0):
    cost = 0.0
    for point_features in points:
      #cost += min (C, np.min ([self.f_distance.pairwise ((cluster_features, point_features))[0, 1] for cluster_features in clusters]))
      cost += np.min ([self.f_distance.pairwise ((cluster_features, point_features))[0, 1] for cluster_features in clusters])
    #for cluster_features in clusters:
    #  cost += np.sum ([min (C, self.f_distance.pairwise ((cluster_features, point_features))[0, 1]) for point_features in points])
    return cost

  """
  Find the cost of assigning data_query to this class for each query_point
  ::::
  data_class    : n_clusters_per_class x n_features
  data_query    : n_queries x n_single_query_points x n_features
  """
  def find_cost (self, data_class, data_query, C=1.0, n_processors = -1):
    def calculate_cost_intern (clusters, points, C=1.0, type_distance = 'euclidean'):
      distance_function = None
      if distance in ['euclidean', 'manhattan', 'hamming', 'canberra', 'braycurtis']:
        distance_function = DistanceMetric.get_metric (distance)
      elif distance in ['chebyshev', 'minkowski', 'wminkowski']:
        distance_function = DistanceMetric.get_metric (distance, params_distance)
      cost = 0.0
      for point_features in points:
        cost += np.min ([distance_function.pairwise((cluster_features, point_features))[0, 1] for cluster_features in clusters])
      return cost
    #total_cost = np.array ([self.calculate_cost (data_class, query_point) for query_point in data_query])
    #return total_cost
    #
    try_parallel = False
    if try_parallel:
      total_cost = Parallel(n_jobs=n_processors) (delayed(calculate_cost_intern)(data_class, data_point_query, C, 'euclidean') for data_point_query in data_query)
      return np.array(tota_cost)
    total_cost = np.empty ([len (data_query)])
    for ith_query in xrange (len (data_query)):
      total_cost[ith_query] = self.calculate_cost (data_class, data_query[ith_query], C)
      #print 'query', ith_query, ':', total_cost
    return np.array(total_cost)

    #total_cost = np.empty ([len (data_query)], np.float)
    #ith_centroid = 0
    #for centroid in data_class:
    #  total_cost[ith_centroid] = np.sum ([self.calculate_cost (clusters, query_point) for query_point in data_query])
    #  ith_cluster += 1
    #return total_cost

  """
  Predict the labels of data_test given data_clusters
  :::::::
  data_test     : n_queries x n_single_query_points x n_features
  """
  def predict (self, data_test, C=1.0, n_processors = 1):
    #all_costs = np.array (Parallel (n_jobs = n_processors) (delayed (find_cost) (self.data_clusters[class_i], data_test, C) for class_i in range (len (self.data_clusters))))

    all_costs = []
    upd_value = 100. / len(self.data_clusters)
    for class_i in xrange (len (self.data_clusters)):
      sys.stdout.write('\r')
      sys.stdout.write("[%-100s] %d%%" % ('='*class_i, upd_value * class_i))
      sys.stdout.flush()
      #print 'assigning to class ', class_i
      all_costs.append (self.find_cost (self.data_clusters[class_i], data_test, C))
    all_costs = np.array (all_costs)

    #all_costs = np.array ([self.find_cost (self.data_clusters[class_i], data_test, C) for class_i in range (len (self.data_clusters))])
    #return all_costs
    for x in all_costs.T:
      print np.array_str (x)
    print '.....'
    return np.array ([self.label[cost.argmin ()] for cost in all_costs.T])

######################################################################################

"""
Find the best k-clusters for the data. This method estimates k
::::
data  : n_points_per_class x n_features_per_point
"""
def find_best_clusters (data, min_k = 2, max_k = 5, thr = 1, n_runs_kmeans = 5):
  return data

  print data.shape
  max_k = max (len (data), max_k)
  data_kmeans = [KMeans (init = 'k-means++', n_clusters = k, n_init = n_runs_kmeans).fit (data) for k in xrange (min_k - 1, max_k + 1)]
  inertia_kmeans = [km.inertia_ for km in data_kmeans]

  for i in xrange (1, len (inertia_kmeans)):
    print i, inertia_kmeans[i]
    if (inertia_kmeans [i - 1] - inertia_kmeans[i] > thr * len (data)):
      return data_kmeans[i].cluster_centers_
  return data_kmeans[-1].cluster_centers_
  print 'Best clusters not found...'
  return None

"""
pensar mejor en como manejar los clusters, por imagenes o por todo el conjunto de imagenes que pertenecen a una clase
"""
