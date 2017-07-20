class Cluster:
    def __init__(self, cluster_time=0, cluster_recency=0):
        self.cluster_time=cluster_time
        self.cluster_recency = cluster_recency
        self.number_of_points = 1