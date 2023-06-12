import torchvision.transforms as ttf
from src.factory import *
from tqdm import tqdm
import torch
import os
import argparse
from src.validate import validate
from scipy.spatial.transform import Rotation as R
import numpy as np
import faiss

msls_cities = {
    'train': ["trondheim", "london", "boston", "melbourne", "amsterdam", "helsinki",
              "tokyo", "toronto", "saopaulo", "moscow", "zurich", "paris", "bangkok",
              "budapest", "austin", "berlin", "ottawa", "phoenix", "goa", "amman", "nairobi", "manila"],
    'val': ["cph", "sf"],
    'test': ["miami", "athens", "buenosaires", "stockholm", "bengaluru", "kampala"]
}


class TestParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()
        self.formatter_class = argparse.ArgumentDefaultsHelpFormatter
        self.add_argument('--dataset', required=True, default='MSLS',
                          help='Name of the dataset [MSLS|7Scenes|TB_Places]')
        self.add_argument('--root_dir', required=True, help='Root directory of the dataset')
        self.add_argument('--subset', required=False, default='val', help='For MSLS. Subset to test')
        self.add_argument('--query_idx_file', type=str, required=False, help='Query idx file, .json')
        self.add_argument('--map_idx_file', type=str, required=False, help='Map idx file, .json')
        self.add_argument('--model_file', type=str, required=True, help='Model file, .pth')
        self.add_argument('--backbone', type=str, default='resnet50',
                          help='which architecture to use. [resnet18, resnet34, resnet50, resnet152, densenet161]')
        self.add_argument('--pool', type=str, required=True, help='pool type', default='avg')
        self.add_argument('--f_length', type=int, default=2048, help='feature length')
        self.add_argument('--image_size', type=str, default="480,640", help='Input size, separated by commas')
        self.add_argument('--norm', type=str, default="L2", help='Normalization descriptors')
        self.add_argument('--batch_size', type=int, default=16, help='Batch size')


def extract_features(dataloader, network, feature_length, features_file):
    if not os.path.exists(features_file):
        feats = np.zeros((len(dataloader.dataset), feature_length))
        for i, batch in tqdm(enumerate(dataloader), desc="Extracting features"):
            x = network.forward(batch.cuda())
            feats[i * dataloader.batch_size:i * dataloader.batch_size + dataloader.batch_size] = x.cpu().detach().squeeze(0)
        np.save(features_file, feats)
    else:
        print(features_file, "already exists. Skipping.")


def extract_features_msls(subset, root_dir, net, f_length, image_t, save_name, results_dir, batch_size, k, class_token=False):
    cities = default_cities[subset]

    result_file = os.path.join(results_dir, f"{save_name}_predictions.txt")
    f = open(result_file, "w+")
    f.close()

    subset_dir = subset if subset == "test" else "train_val"
    for city in cities:
        print(city)
        query_index_file = os.path.join(root_dir, subset_dir, city, "query.json")
        query_dataloader = create_dataloader("test", root_dir, query_index_file, None, image_t, batch_size)
        query_features_file = os.path.join(results_dir, f"{save_name}_{city}_queryfeats.npy")
        if class_token: extract_features(query_dataloader, net, f_length, query_features_file)
        else: extract_features(query_dataloader, net, f_length, query_features_file)

        map_index_file = os.path.join(root_dir, subset_dir, city, "database.json")
        map_dataloader = create_dataloader("test", root_dir, map_index_file, None, image_t, batch_size)
        map_features_file = os.path.join(results_dir, f"{save_name}_{city}_mapfeats.npy")
        if class_token: extract_features(map_dataloader, net, f_length, map_features_file)
        else: extract_features(map_dataloader, net, f_length, map_features_file)

        result_file = extract_msls_top_k(map_features_file, query_features_file, map_index_file, query_index_file, result_file, k)

    if subset == "val":
        print(result_file)
        score_file = result_file.replace("_predictions", "_result")
        if not os.path.exists(score_file):
            validate(result_file, root_dir, score_file)


def load_index(index):
    with open(index) as file:
        data = json.load(file)
    image_paths = np.array(data["im_paths"])
    image_prefix = data["im_prefix"]

    if "poses" in data.keys():
        poses = np.array(data["poses"])
        return image_paths, poses, image_prefix
    else:
        return image_paths, image_prefix


def world_to_camera(pose):
    [w_qw, w_qx, w_qy, w_qz, w_tx, w_ty, w_tz] = pose
    r = R.from_quat([w_qx, w_qy, w_qz, w_qw]).as_matrix().T
    tx, ty, tz = np.dot(np.array([w_tx, w_ty, w_tz]), np.linalg.inv(-r))
    qx, qy, qz, qw = R.from_matrix(r).as_quat()
    return qw, qx, qy, qz, tx, ty, tz


def predict_poses_cmu(root_dir, map_features_file, query_features_file):
    _, reference_poses, _ = load_index(os.path.join(root_dir, "reference.json"))
    test_images, _ = load_index(os.path.join(root_dir, "test.json"))
    _, best_score = search(map_features_file, query_features_file, 1)

    name = "ExtendedCMU" if "extended" in map_features_file else "CMU"
    name = map_features_file \
        .replace("_mapfeats", "_toeval") \
        .replace("/MSLS_", f"/{name}_eval_MSLS_") \
        .replace(".npy", ".txt")

    with open(name, "w") as f:
        for q, db_index in tqdm(zip(test_images, best_score), desc="Predicting poses..."):
            cut_place = q.find("/img")
            query_image_to_submit = q[cut_place + 1:]
            pose = np.array((reference_poses[db_index])).flatten()
            submission = f"{query_image_to_submit} {' '.join(pose.astype(str))}\n"
            f.write(submission)


def predict_poses(root_dir, map_features_file, query_features_file):
    _, reference_poses, _ = load_index(os.path.join(root_dir, "reference.json"))
    test_images, _ = load_index(os.path.join(root_dir, "test.json"))
    _, best_score = search(map_features_file, query_features_file, 1)

    name = map_features_file \
        .replace("_mapfeats", "_toeval") \
        .replace("/MSLS_", "/RobotCar_eval_MSLS_") \
        .replace(".npy", ".txt")

    with open(name, "w") as file:
        for query_image, database_index in tqdm(zip(test_images, best_score), desc="Predicting poses..."):
            cut_place = query_image.find("/rear")
            query_image_to_submit = query_image[cut_place + 1:]
            assert query_image_to_submit.startswith("rear/")
            pose = np.array(world_to_camera(reference_poses[database_index].flatten()))
            submission = f"{query_image_to_submit} {' '.join(pose.astype(str))}\n"
            file.write(submission)


def eval_pitts(root_dir, dataset, result_file):
    if "pitts" in dataset:
        gt_file = os.path.join(root_dir, f"{dataset}_test_gt.h5")
    elif dataset.lower() == "tokyotm":
        gt_file = os.path.join(root_dir, "val_gt.h5")
    else:
        gt_file = os.path.join(root_dir, "gt.h5")
    results_index = np.load(result_file)
    score_file = result_file.replace("predictions.npy", "scores.txt")
    print(results_index.shape)
    ks = [1, 2, 3, 4, 5, 10, 15, 20, 25]
    with open(score_file, "w") as sf:
        with h5py.File(gt_file, "r") as f:
            gt = f["sim"]
            print(gt.shape)
            for k in ks:
                hits = 0
                total = 0
                for query_index, ret in enumerate(results_index):
                    if np.any(gt[query_index, :]):
                        total += 1
                        database_index = sorted(ret[:k])
                        hits += np.any(gt[query_index, database_index])
                print(k, np.round(hits / total * 100, 2))
                sf.write(f"{k},{(np.round(hits / total * 100, 2))}\n")


def extract_features_map_query(root_dir, query_index_file, map_index_file, network, feature_length, save_name, results_dir, batch_size, k, dataset):
    query_dataloader = create_dataloader("test", root_dir, query_index_file, None, transformer, batch_size)
    query_features_file = os.path.join(results_dir, f"{save_name}_queryfeats.npy")
    extract_features(query_dataloader, network, feature_length, query_features_file)

    map_dataloader = create_dataloader("test", root_dir, map_index_file, None, transformer, batch_size)
    map_features_file = os.path.join(results_dir, f"{save_name}_mapfeats.npy")
    extract_features(map_dataloader, network, feature_length, map_features_file)

    result_file = os.path.join(results_dir, f"{save_name}_predictions.npy")

    if dataset.lower() == "tokyotm":
        extract_top_k_tokyotm(map_features_file, query_features_file, map_index_file, query_index_file, result_file, k)
    else:
        extract_top_k(map_features_file, query_features_file, result_file, k)
    if dataset == "robotcarseasons":
        predict_poses(root_dir, map_features_file, query_features_file)
    elif dataset == "extendedcmu" or dataset == "cmu":
        predict_poses_cmu(root_dir, map_features_file, query_features_file)
    elif "pitts" in dataset or "tokyo" in dataset:
        eval_pitts(root_dir, dataset, result_file)


def extract_top_k_tokyotm(map_features_file, query_features_file, database_index_file, query_index_file, result_index_file, k):
    print("TokyoTM")
    D, best_score = search(map_features_file, query_features_file)
    with open(database_index_file, "r") as f:
        db_paths = np.array(json.load(f)["im_paths"])
    with open(query_index_file, "r") as f:
        query_paths = np.array(json.load(f)["im_paths"])
    result_idx = np.zeros((len(query_paths), k))
    for index, query in enumerate(query_paths):
        query_timestamp = int(query.split("/")[3][1:])
        aux = 0
        for t in range(k):
            score = best_score[index, aux]
            database = db_paths[score]
            database_timestamp = int(database.split("/")[3][1:])

            # ensure we retrieve something at least a month away
            while np.abs(query_timestamp - database_timestamp) < 1:
                aux += 1
                score = best_score[index, aux]
                database = db_paths[score]
                database_timestamp = int(database.split("/")[3][1:])
            result_idx[index, t] = best_score[index, aux]
            aux += 1

    np.save(result_index_file, result_idx.astype(int))


def extract_msls_top_k(map_feats_file, query_feats_file, database_index_file, query_index_file, result_file, k):
    D, I = search(map_feats_file, query_feats_file, k)

    # load indices
    with open(database_index_file, "r") as file:
        db_paths = np.array(json.load(file)["im_paths"])
    with open(query_index_file, "r") as file:
        q_paths = np.array(json.load(file)["im_paths"])
    with open(result_file, "a+") as file:
        for i, query in enumerate(q_paths):
            query_id = query.split("/")[-1].split(".")[0]
            db_paths_str = ' '.join([db_paths[j].split('/')[-1].split('.')[0] for j in I[i, :]])
            file.write(f"{query_id} {db_paths_str}\n")
    return result_file


def search(map_feats_file, query_feats_file, k=25):
    # load features
    query_features = np.load(query_feats_file).astype('float32')
    map_features = np.load(map_feats_file).astype('float32')
    k = k or map_features.shape[0]
    # build index and add map features
    index = faiss.IndexFlatL2(map_features.shape[1])
    index.add(map_features)
    # search top K
    D, I = index.search(query_features.astype('float32'), k)
    return D, I


def extract_top_k(map_feats_file, query_feats_file, result_file, k):
    D, I = search(map_feats_file, query_feats_file, k)
    np.save(result_file, I)


def image_transformer(image_size):
    if len(image_size) == 2:
        print("testing with images of size", image_size[0], image_size[1])
        return ttf.Compose([
            ttf.Resize(size=(image_size[0], image_size[1])),
            ttf.ToTensor(),
            ttf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        print("testing with images of size", image_size[0])
        return ttf.Compose([
            ttf.Resize(size=image_size[0]),
            ttf.ToTensor(),
            ttf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


if __name__ == "__main__":
    parser = TestParser()
    params = parser.parse_args()

    # Create model and load weights
    pool = params.pool
    test_net = create_model(params.backbone, pool, norm=params.norm, mode="single")
    try: test_net.load_state_dict(torch.load(params.model_file)["model_state_dict"])
    finally: test_net.load_state_dict(torch.load(params.model_file)["state_dict"])

    # Evaluate model
    if torch.cuda.is_available(): test_net.cuda()
    test_net.eval()

    # Create the datasets
    image_size = [int(x) for x in params.image_size.split(",")]
    transformer = image_transformer(image_size)
    feature_length = int(params.f_length)

    results_dir = os.path.join("results", params.dataset, params.subset)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    save_name = params.model_file.split("/")[-1].split(".")[0]
    print(save_name)

    if params.dataset.lower() == "msls":
        extract_features_msls(params.subset, params.root_dir, test_net, feature_length, transformer, save_name,
                              results_dir, params.batch_size, 30)
    else:
        extract_features_map_query(params.root_dir, params.query_idx_file, params.map_idx_file, test_net,
                                   feature_length, save_name, results_dir, params.batch_size, 30,
                                   params.dataset.lower())
