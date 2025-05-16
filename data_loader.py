import numpy as np
import pandas as pd
import os
import networkx as nx


# Integrate ANOVA + PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.validation import check_is_fitted

from utils import normalize_adjencency_mat, convert_scipy_CRS_space_to_tensor
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force TensorFlow to use CPU

class EllipticDatasetLoader(object):
    def __init__(self, datadir_path, test_portion=0.3, filter_unknown=False, local_features_only=True,
                 use_scaler=True, use_anova=False, n_anova_features=50, use_pca=True, n_pca_components=30,
                 use_svd=False, n_svd_components=30):
        
        self.filter_unknown = filter_unknown

        classes_csv = os.path.join(datadir_path, 'elliptic_txs_classes.csv')
        edgelist_csv = os.path.join(datadir_path, 'elliptic_txs_edgelist.csv')
        features_csv = os.path.join(datadir_path, 'elliptic_txs_features.csv')

        classes = pd.read_csv(classes_csv, index_col='txId')  # labels are 'unknown', '1'(illicit), '2'
        edgelist = pd.read_csv(edgelist_csv)
        features = pd.read_csv(features_csv, header=None, index_col=0)  # features of the transactions
        data = pd.concat([classes,features],axis=1)

        # ======================================================
        # STEP 1: Extract labeled features for ANOVA, PCA or SVD
        # ======================================================

        #set up transformers
        self.use_scaler = use_scaler
        self.use_anova = use_anova
        self.use_pca = use_pca
        self.use_svd = use_svd

        self.scaler = StandardScaler() if use_scaler else None
        self.anova = SelectKBest(score_func=f_classif, k=n_anova_features) if use_anova else None
        self.pca = PCA(n_components=n_pca_components) if use_pca else None
        self.svd = TruncatedSVD(n_components=n_svd_components) if use_svd else None


        labeled_data = data[data['class'].isin(['1', '2'])].copy()
        class_map = {'1': 0, '2': 1}
        labels = labeled_data['class'].map(class_map).values
        all_features = labeled_data.iloc[:, 2:].values.astype(np.float32)  # All 167 features

        # Optional: select only 94 local features if local_features_only is True
        if local_features_only:
            all_features = all_features[:, :94]

        X = all_features

        if self.scaler:
            X = self.scaler.fit_transform(X)
        if self.anova:
            X = self.anova.fit_transform(X, labels)
        if self.pca:
            X = self.pca.fit(X)
        if self.svd:
            X = self.svd.fit(X)

        print("✅ Feature pipeline summary:")
        if self.scaler: print(" - StandardScaler: applied")
        if self.anova: print(f" - ANOVA: top {self.anova.k} features selected")
        if self.pca: print(f" - PCA: {self.pca.n_components} components retained")
        if self.svd: print(f" - SVD: {self.svd.n_components} components retained")

        # 🔍 DEBUG START — Inspect transformations
        if self.scaler:
            scaled_feats = self.scaler.transform(all_features)
            print("📊 Scaled feature shape:", scaled_feats.shape)
            # print("📊 Sample scaled feature [0]:")
            # print(scaled_feats[0])

        if self.anova:
            anova_scores = self.anova.scores_
            selected_indices = self.anova.get_support(indices=True)
            print("🎯 ANOVA F-scores (first 10):", anova_scores[:10])
            print("🎯 Selected feature indices:", selected_indices)

        if self.pca:
            print("🧬 PCA explained variance ratio (first 5):", self.pca.explained_variance_ratio_[:5])
            print("🧬 PCA components shape:", self.pca.components_.shape)
            # print("🧬 PCA components (first row):")
            # print(self.pca.components_[0])
        elif self.svd:
            print("🧪 SVD explained variance ratio (first 5):", self.svd.explained_variance_ratio_[:5])
            print("🧪 SVD components shape:", self.svd.components_.shape)
            # print("🧪 SVD components (first row):")
            # print(self.svd.components_[0])
        # 🔍 DEBUG END

        
        num_features = features.shape[1]
        timesteps = np.unique(features[1])

        graph = nx.DiGraph()
        feature_idx = [i+2 for i in range(93+72)]
        if local_features_only:
            feature_idx = feature_idx[:94]
        for tx_idx, features in data.iterrows():
            graph.add_node(tx_idx,label=str(features["class"]),timestamp=features[1],features=features[feature_idx])
        graph.add_edges_from(edgelist.values)

        train_timesteps, test_timesteps = np.split(timesteps,[int(timesteps.shape[0]*(1-test_portion))])
        self.train_graphs = []
        self.train_triples = []
        self.test_graphs = []
        self.test_triples = []
        one_hot = np.eye(2,dtype=np.float32) if self.filter_unknown else np.eye(3,dtype=np.float32)
        class_converter = {"1":0, "2":1, "unknown":2}
        for comp in nx.weakly_connected_components(graph):
            sg = graph.subgraph(comp)
            if self.filter_unknown:
                sg.remove_nodes_from(classes[classes["class"] == "unknown"].index.tolist())

            ts = np.unique([d for _,d in sg.nodes(data="timestamp")])
            if ts.shape[0] > 1:
                raise RuntimeError("incorrect division on timestamps")

            nodes, targets = [], []
            for _, d in sg.nodes(data=True):
                targets.append(class_converter[d["label"]])
                
                '''nodes.append(d["features"].values.astype(np.float32))'''
                # ---------------------------------------------------------
                raw_feat = d["features"].values.astype(np.float32)

                # Optional: select only local features
                if local_features_only:
                    raw_feat = raw_feat[:94]
                # print(f"Raw features for node {d}: {raw_feat}")

                x = raw_feat.reshape(1, -1)

                if self.scaler:
                    x = self.scaler.transform(x)
                if self.anova:
                    x = self.anova.transform(x)
                if self.pca:
                    x = self.pca.transform(x)
                elif self.svd:
                    x = self.svd.transform(x)

                nodes.append(x.squeeze(0))
                # ---------------------------------------------------------

            nodes = np.vstack(nodes)
            targets = one_hot[np.array(targets)]
            adjacency_mat = normalize_adjencency_mat(nx.adjacency_matrix(sg).astype(np.float32))
            adjacency_mat = convert_scipy_CRS_space_to_tensor(adjacency_mat)

            if ts[0] in train_timesteps:
                self.train_graphs.append(sg)
                self.train_triples.append((nodes,targets, adjacency_mat))
            else:
                self.test_graphs.append(sg)
                self.test_triples.append((nodes, targets, adjacency_mat))

        if len(self.train_graphs) + len(self.test_graphs) != timesteps.shape[0]:
            raise RuntimeError("number of generated graphs goes not match number of timestamps")

    @property
    def num_classes(self):
        return 3

    def test_batch_iterator(self):
        for i in range(len(self.test_graphs)):
            g = self.test_graphs[i]
            n, t, adj = self.test_triples[i]
            yield g, n, t, adj

    def train_batch_iterator(self):
        idx = np.arange(len(self.train_graphs))
        np.random.shuffle(idx)
        for i in idx:
            g = self.train_graphs[i]
            n, t, adj = self.train_triples[i]
            yield g, n, t, adj


if __name__=="__main__":
    DATADIR = "elliptic_bitcoin_dataset"
    FILTER_UNKNOWN = False
    ONLY_LOCAL_FEATURE = False
    CLASS_WEIGTHS = [0.7,0.29,0.01]
    TEST_SHARE = 0.3
    NUM_EPOCH = 500
    LEARNING_RATE = 1e-3

    dl = EllipticDatasetLoader(DATADIR, TEST_SHARE, FILTER_UNKNOWN,local_features_only=ONLY_LOCAL_FEATURE)
    print(dl)
