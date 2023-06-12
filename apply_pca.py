from tqdm import tqdm
import torch
import os
import numpy as np
import argparse
from src.validate import validate
from extract_predictions import \
    extract_msls_top_k, predict_poses, predict_poses_cmu, eval_pitts, extract_top_k, extract_top_k_tokyotm

msls_cities = {
    'train': ["trondheim", "london", "boston", "melbourne", "amsterdam", "helsinki",
              "tokyo", "toronto", "saopaulo", "moscow", "zurich", "paris", "bangkok",
              "budapest", "austin", "berlin", "ottawa", "phoenix", "goa", "amman", "nairobi", "manila"],
    'val': ["cph", "sf"],
    'test': ["miami", "athens", "buenosaires", "stockholm", "bengaluru", "kampala"]
}


# Whitening code by Filip Radenovic
# https://github.com/filipradenovic/cnnimageretrieval-pytorch/blob/master/cirtorch/utils/whiten.py
def cholesky(S):
    # Cholesky decomposition with adding a small value on the diagonal until matrix is positive definite
    alpha = 0
    while 1:
        try:
            L = np.linalg.cholesky(S + alpha * np.eye(*S.shape))
            return L
        finally:
            if alpha == 0:
                alpha = 1e-10
            else:
                alpha *= 10
            print(">>>> {}::cholesky: Matrix is not positive definite, adding {:.0e} on the diagonal"
                  .format(os.path.basename(__file__), alpha))


def whiten_apply(X, m, P, dimensions=None):
    if not dimensions:
        dimensions = P.shape[0]
    X = np.dot(P[:dimensions, :], X - m)
    X = X / (np.linalg.norm(X, ord=2, axis=0, keepdims=True) + 1e-6)
    return X.copy(order='C')


def pca_whiten_learn(X):
    N = X.shape[1]

    # Learning PCA w/o annotations
    m = X.mean(axis=1, keepdims=True)
    Xc = X - m
    X_cov = np.dot(Xc, Xc.T)
    X_cov = (X_cov + X_cov.T) / (2 * N)
    eigenvalue, eigenvector = np.linalg.eig(X_cov)
    order = eigenvalue.argsort()[::-1]
    eigenvalue = eigenvalue[order]
    eigenvector = eigenvector[:, order]

    P = np.dot(np.linalg.inv(np.sqrt(np.diag(eigenvalue))), eigenvector.T)
    if P.dtype == "complex128":
        P = np.real(P)
        print("Warning: complex numbers in eigenvector and eigenvalues")
    return m, P


def map_query_pca_whiten_learn(params):
    db = np.load(params.map_feats_file).astype(np.float16).T
    return pca_whiten_learn(db)


def map_query_whiten_apply(dataset, name, root_dir, subset, map_feats_file, query_feats_file, m, P, map_index_file="",
                           query_index_file="", m_raw_file="", dimensions=None):
    dimensions = dimensions or [2048, 1024, 512, 256, 128, 64, 32]
    features_dir = os.path.join("results", dataset, subset)
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)
    database = np.load(map_feats_file).astype(np.float16).T
    query = np.load(query_feats_file).astype(np.float16).T
    for dim in tqdm(dimensions, desc="Applying PCA whitening..."):
        query_whiten_file = query_feats_file.replace(".npy", f"_whiten_{dim}.npy")
        if not os.path.exists(query_whiten_file):
            print("Getting query features...")
            query_whiten = whiten_apply(query, m, P, dimensions=dim).T
            query_whiten = query_whiten.copy(order='C')
            np.save(query_whiten_file, query_whiten)
        else:
            print("Loading query features...")
            query_whiten = np.load(query_whiten_file)

        database_whiten_file = map_feats_file.replace(".npy", f"_whiten_{dim}.npy")
        if not os.path.exists(database_whiten_file):
            print("Getting map features...")
            database_whiten = whiten_apply(database, m, P, dimensions=dim).T
            database_whiten = database_whiten.copy(order='C')
            np.save(database_whiten_file, database_whiten)
        else:
            print("Loading map features...")
            database_whiten = np.load(database_whiten_file)

        if dataset.lower() == "robotcarseasons":
            predict_poses(root_dir, database_whiten_file, query_whiten_file)
        elif dataset.lower() == "extendedcmu" or dataset.lower() == "cmu":
            predict_poses_cmu(root_dir, database_whiten_file, query_whiten_file)
        elif "pitts" in dataset.lower() or dataset.lower() == "tokyo247":
            result_file = database_whiten_file.replace("_mapfeats", "").replace(".npy", "_predictions.npy")
            extract_top_k(database_whiten_file, query_whiten_file, result_file, 30)
            eval_pitts(root_dir, dataset, result_file)
        elif dataset.lower() == "tokyotm":
            result_file = database_whiten_file.replace("_mapfeats", "").replace(".npy", "_predictions.npy")
            map_index_file = os.path.join(root_dir, "val_db.json")
            query_index_file = os.path.join(root_dir, "val_q.json")
            extract_top_k_tokyotm(database_whiten_file, query_whiten_file, map_index_file, query_index_file, result_file, 50)
            eval_pitts(root_dir, dataset, result_file)
        elif dataset.lower() == "msls":
            result_file = os.path.join(features_dir, f"{name}_retrieved_whiten_{dim}.csv")
            extract_msls_top_k(database_whiten_file, query_whiten_file, map_index_file, query_index_file, result_file, 50)


def map_query_whiten_apply_from_file(dataset, name, root_dir, subset, dim, map_feats_file, query_feats_file,
                                     checkpoint_file, m_idx_file="", q_idx_file="", m_raw_file=""):
    checkpoint = torch.load(f"{checkpoint_file}{dim}.pth")
    assert dim == checkpoint["num_pcs"]
    pca_conv = torch.nn.Conv2d(32768, dim, kernel_size=(1, 1), stride=1, padding=0)
    pca_conv.weight = torch.nn.Parameter(checkpoint["state_dict"]["WPCA.0.weight"].to(torch.float32))
    pca_conv.bias = torch.nn.Parameter(checkpoint["state_dict"]["WPCA.0.bias"].to(torch.float32))
    features_dir = os.path.join("results", dataset, subset)
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)
    with torch.no_grad():
        b_size = 1024
        q_whiten_file = query_feats_file.replace(".npy", f"_whiten_{dim}.npy")
        if not os.path.exists(q_whiten_file):
            print("Getting query features...")
            aux = 0
            q = torch.Tensor(np.load(query_feats_file))
            w_q_feats = torch.zeros((q.shape[0], dim))
            while aux < q.shape[0]:
                print(aux)
                w_q_feats[aux: aux + b_size, :] = torch.nn.functional.normalize(
                    pca_conv(q[aux:aux + b_size, :].unsqueeze(-1).unsqueeze(-1)).squeeze(), p=2, dim=-1)
                aux = aux + b_size
            np.save(q_whiten_file, w_q_feats.detach().numpy())
        else:
            print(q_whiten_file, "already exists. Skipping...")

        db_whiten_file = map_feats_file.replace(".npy", f"_whiten_{dim}.npy")
        if not os.path.exists(db_whiten_file):
            print("Getting map features...")
            db = torch.Tensor(np.load(map_feats_file))
            aux = 0
            w_m_feats = torch.zeros((db.shape[0], dim))
            while aux < db.shape[0]:
                print(aux)
                w_m_feats[aux:aux + b_size, :] = torch.nn.functional.normalize(
                    pca_conv(db[aux: aux + b_size, :].unsqueeze(-1).unsqueeze(-1)).squeeze(), p=2, dim=-1)
                aux = aux + b_size
            np.save(db_whiten_file, w_m_feats.detach().numpy())
        else:
            print(db_whiten_file, "already exists. Skipping...")
    if dataset.lower() == "robotcarseasons":
        predict_poses(root_dir, db_whiten_file, q_whiten_file)
    elif dataset.lower() == "extendedcmu" or dataset.lower() == "cmu":
        predict_poses_cmu(root_dir, db_whiten_file, q_whiten_file)
    elif "pitts" in dataset.lower() or dataset.lower() == "tokyo247":
        result_file = db_whiten_file.replace("_mapfeats", "").replace(".npy", "_predictions.npy")
        extract_top_k(db_whiten_file, q_whiten_file, result_file, 30)
        eval_pitts(root_dir, dataset, result_file)
    elif dataset.lower() == "tokyotm":
        result_file = db_whiten_file.replace("_mapfeats", "").replace(".npy", "_predictions.npy")
        m_idx_file = os.path.join(root_dir, "val_db.json")
        q_idx_file = os.path.join(root_dir, "val_q.json")
        extract_top_k_tokyotm(db_whiten_file, q_whiten_file, m_idx_file, q_idx_file, result_file, 50)
        eval_pitts(root_dir, dataset, result_file)
    elif dataset.lower() == "msls":
        result_file = os.path.join(features_dir, f"{name}_retrieved_whiten_{dim}.csv")
        extract_msls_top_k(db_whiten_file, q_whiten_file, m_idx_file, q_idx_file, result_file, 50)


def msls_pca_whiten_learn(dataset, subset, name):
    features_dir = os.path.join("results", dataset, subset)
    cities = msls_cities[subset]
    db = []
    for city in cities:
        db_file = os.path.join(features_dir, f"{name}_{city}_mapfeats.npy")
        db.append(np.load(db_file).T)
    db = np.hstack(db)
    return pca_whiten_learn(db)


def msls_whiten_apply_from_file(root_dir, dataset, subset, name, dim, checkpoint):
    cities = msls_cities[subset]
    features_dir = os.path.join("results", dataset, subset)
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)
    result_file = os.path.join(features_dir, f"{name}_retrieved_whiten_{dim}.csv")
    f = open(result_file, "w+")
    f.close()
    for city in tqdm(cities):
        database_file = os.path.join(features_dir, f"{name}_{city}_mapfeats.npy")
        query_file = os.path.join(features_dir, f"{name}_{city}_queryfeats.npy")
        database_folder = subset if subset == "test" else "train_val"
        query_index_file = os.path.join(root_dir, database_folder, city, "query.json")
        map_index_file = os.path.join(root_dir, database_folder, city, "database.json")
        map_raw_file = os.path.join(root_dir, database_folder, city, "database", "raw.csv")

        map_query_whiten_apply_from_file(dataset, name, root_dir, subset, dim,
                                         database_file, query_file, checkpoint, m_idx_file=map_index_file,
                                         q_idx_file=query_index_file, m_raw_file=map_raw_file)
    if subset == "val":
        result_file = os.path.join(features_dir, f"{name}_retrieved_whiten_{dim}.csv")
        validate(result_file, root_dir, result_file.replace("retrieved", "result").replace(".csv", ".txt"))


def msls_whiten_apply(root_dir, dataset, subset, name, m, P, dimensions=None):
    dimensions = dimensions or [2048, 1024, 512, 256, 128, 64, 32]
    cities = msls_cities[subset]
    features_dir = os.path.join("results", dataset, subset)
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)
    for dim in dimensions:
        result_file = os.path.join(features_dir, f"{name}_retrieved_whiten_{dim}.csv")
        f = open(result_file, "w+")
        f.close()
    for city in cities:
        db_file = os.path.join(features_dir, f"name_{city}_mapfeats.npy")
        q_file = os.path.join(features_dir, f"name_{city}_queryfeats.npy")
        ds_folder = subset if subset == "test" else "train_val"
        q_idx_file = os.path.join(root_dir, ds_folder, city, "query.json")
        m_idx_file = os.path.join(root_dir, ds_folder, city, "database.json")
        m_raw_file = os.path.join(root_dir, ds_folder, city, "database", "raw.csv")

        map_query_whiten_apply(dataset, name, root_dir, subset, db_file, q_file, m, P,map_index_file=m_idx_file,
                               query_index_file=q_idx_file, m_raw_file=m_raw_file, dimensions=dimensions)

    if subset == "val":
        for dim in tqdm(dimensions):
            result_file = os.path.join(features_dir, f"{name}_retrieved_whiten_{dim}.csv")
            validate(result_file, root_dir, result_file.replace("retrieved", "result").replace(".csv", ".txt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', required=True, default='MSLS', help='Name of the dataset [MSLS|7Scenes|TB_Places]')
    parser.add_argument('--root_dir', required=True, help='Root directory of the dataset')
    parser.add_argument('--subset', required=False, default='val', help='For MSLS. Subset to test')
    parser.add_argument('--checkpoint', required=False, default=None, help='Checkpoint containing the PCs')
    parser.add_argument('--query_feats_file', type=str, required=False, help='Query features file, .npy')
    parser.add_argument('--map_feats_file', type=str, required=False, help='Map features file, .npy')
    parser.add_argument('--name', type=str, required=True, help='Name of the experiment')
    parser.add_argument('--dim', type=int, required=False, help='dimension size')
    params = parser.parse_args()

    if "vgg" in params.name:
        dimensions = [512, 256, 128, 64, 32]
    elif "vlad" in params.name.lower():
        dimensions = [32768, 16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32]
    else:
        dimensions = [2048, 1024, 512, 256, 128, 64, 32]

    if params.dim is not None:
        dimensions = [params.dim]

    if params.dataset == "MSLS":
        if params.checkpoint is None:
            m, P = msls_pca_whiten_learn(params.dataset, params.subset, params.name)
            msls_whiten_apply(params.root_dir, params.dataset, params.subset, params.name, m, P, dimensions=dimensions)
        else:
            msls_whiten_apply_from_file(params.root_dir, params.dataset, params.subset, params.name, params.dim,
                                        params.checkpoint)
    else:
        if params.checkpoint is None:
            m, P = map_query_pca_whiten_learn(params)
            map_query_whiten_apply(params.dataset, params.name, params.root_dir, params.subset, params.map_feats_file,
                                   params.query_feats_file, m, P, dimensions=dimensions)
        else:
            map_query_whiten_apply_from_file(params.dataset, params.name, params.root_dir, params.subset, params.dim,
                                             params.map_feats_file, params.query_feats_file, params.checkpoint)
