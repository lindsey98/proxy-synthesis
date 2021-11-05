'''
proxy-synthesis
Copyright (c) 2021-present NAVER Corp.
Apache License v2.0
'''
import faiss
import numpy as np
import sklearn.cluster
import torch
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
import sklearn
from utils.map import *

def evaluation(X, Y, Kset, args):

    def evaluation_faiss(X, Y, Kset, args):
        if args.data_name.lower() != 'inshop':
            kmax = np.max(Kset + [args.max_r])  # search K
        else:
            kmax = np.max(Kset)

        test_class_dict = args.test_class_dict

        # compute NMI
        if args.do_nmi:
            classN = np.max(Y) + 1
            kmeans = faiss.Kmeans(d=X.shape[1], k=int(classN))
            kmeans.train(X)
            labels = kmeans.index.search(X, 1)[1]
            labels = np.squeeze(labels, 1)
            nmi = normalized_mutual_info_score(Y, labels, average_method='arithmetic')
        else:
            nmi = 0.0
        print(nmi)

        if args.data_name.lower() != 'inshop':
            offset = 1
            X_query = X
            X_gallery = X
            Y_query = Y
            Y_gallery = Y

        else:  # inshop
            offset = 0
            len_gallery = len(args.gallery_labels)
            X_gallery = X[:len_gallery, :]
            X_query = X[len_gallery:, :]
            Y_query = args.query_labels
            Y_gallery = args.gallery_labels

        nq, d = X_query.shape
        ng, d = X_gallery.shape
        I = np.empty([nq, kmax + offset], dtype='int64')
        D = np.empty([nq, kmax + offset], dtype='float32')

        if hasattr(faiss, 'StandardGpuResources'):
            res = faiss.StandardGpuResources()
            flat_config = faiss.GpuIndexFlatConfig()
            flat_config.device = 0

            max_k = max(Kset)
            index_function = faiss.GpuIndexFlatIP
            index = index_function(res, X_gallery.shape[1], flat_config)
        else:
            max_k = max(Kset)
            index_function = faiss.IndexFlatIP
            index = index_function(X_gallery.shape[1])

        index.add(X_gallery)
        closest_indices = index.search(X_query, max_k + offset)[1]
        recalls = {}
        for i, k in enumerate(Kset):
            indices = closest_indices[:, offset:k + offset]
            recalls[i] = (Y_query[:, None] == Y_gallery[indices]).any(1).mean()

        print(recalls)
        indices = closest_indices[:, offset:]
        YNN = Y_gallery[indices]

        if args.data_name.lower() != 'inshop':
            label_counts = get_label_match_counts(torch.from_numpy(Y_query), torch.from_numpy(Y_query))  # get R
            num_k = max([count[1] for count in label_counts])
            knn_indices = get_knn(
                torch.from_numpy(X_query), torch.from_numpy(X_query), num_k, True
            )

            knn_labels = torch.from_numpy(Y_query)[knn_indices]  # get KNN indicies
            map_R = mean_average_precision_at_r(knn_labels=knn_labels,
                                                gt_labels=torch.from_numpy(Y_query)[:, None],
                                                embeddings_come_from_same_source=True,
                                                label_counts=label_counts,
                                                avg_of_avgs=False,
                                                label_comparison_fn=torch.eq)
            print("MAP@R:{:.3f}".format(map_R * 100))

        else:  # inshop
            label_counts = get_label_match_counts(Y_query, Y_gallery)  # get R
            num_k = max([count[1] for count in label_counts])

            knn_indices = get_knn(
                X_gallery, X_query, num_k, True
            )
            knn_labels = Y_gallery[knn_indices]  # get KNN indicies
            map_R = mean_average_precision_at_r(knn_labels=knn_labels,
                                                gt_labels=Y_query[:, None],
                                                embeddings_come_from_same_source=False,
                                                label_counts=label_counts,
                                                avg_of_avgs=False,
                                                label_comparison_fn=torch.eq)
            print("MAP@R:{:.3f}".format(map_R * 100))

        return nmi, recalls, map_R

    return evaluation_faiss(X, Y, Kset, args)