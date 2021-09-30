'''
proxy-synthesis
Copyright (c) 2021-present NAVER Corp.
Apache License v2.0
'''
import faiss
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score


def evaluation(X, Y, Kset, args):
    # def get_recallK(Y_query, YNN, Kset):
    #     recallK = np.zeros(len(Kset))
    #     num = Y_query.shape[0]
    #     for i in range(0, len(Kset)):
    #         pos = 0.
    #         for j in range(0, num):
    #             if Y_query[j] in YNN[j, :Kset[i]]:
    #                 pos += 1.
    #         recallK[i] = pos / num
    #     return recallK

    def get_Rstat(Y_query, YNN, test_class_dict):
        '''
        test_class_dict:
            key = class_idx, value = the number of images
        '''
        RP_list = []
        MAP_list = []

        for gt, knn in zip(Y_query, YNN):
            n_imgs = test_class_dict[gt] - 1  # - 1 for query.
            selected_knn = knn[:n_imgs]
            correct_array = (selected_knn == gt).astype('float32')

            RP = np.mean(correct_array)

            MAP = 0.0
            sum_correct = 0.0
            for idx, correct in enumerate(correct_array):
                if correct == 1.0:
                    sum_correct += 1.0
                    MAP += sum_correct / (idx + 1.0)
            MAP = MAP / n_imgs

            RP_list.append(RP)
            MAP_list.append(MAP)

        return np.mean(RP_list), np.mean(MAP_list)

    def evaluation_faiss(X, Y, Kset, args):
        if args.data_name.lower() != 'inshop':
            kmax = np.max(Kset + [args.max_r])  # search K
        else:
            kmax = np.max(Kset)

        test_class_dict = args.test_class_dict

        # compute NMI
        if args.do_nmi:
            classN = np.max(Y) + 1
            kmeans = KMeans(n_clusters=classN).fit(X)
            nmi = normalized_mutual_info_score(Y, kmeans.labels_, average_method='arithmetic')
        else:
            nmi = 0.0

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
        # res = faiss.StandardGpuResources()
        # res.setDefaultNullStreamAllDevices()
        # faiss.bruteForceKnn(res, faiss.METRIC_INNER_PRODUCT,
        #                     faiss.swig_ptr(X_gallery), True, ng,
        #                     faiss.swig_ptr(X_query), True, nq,
        #                     d, int(kmax + offset),
        #                     faiss.swig_ptr(D),
        #                     faiss.swig_ptr(I))
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

        indices = closest_indices[:, offset:]
        YNN = Y_gallery[indices]
        # recallK = get_recallK(Y_query, YNN, Kset)

        if args.data_name.lower() != 'inshop':
            RP, MAP = get_Rstat(Y_query, YNN, test_class_dict)
        else:  # inshop
            RP = 0
            MAP = 0

        return nmi, recalls, RP, MAP

    return evaluation_faiss(X, Y, Kset, args)