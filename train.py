import argparse
import shutil

import torchvision.transforms as ttf
from torch.optim.lr_scheduler import StepLR
from torch import optim
from extract_predictions import extract_msls_top_k

from src.validate import validate as msls_validate
from src.factory import *
from src.criteria import *


class TrainParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()
        self.formatter_class = argparse.ArgumentDefaultsHelpFormatter
        self.add_argument('--root_dir', required=True, help='Root directory of the dataset')
        self.add_argument('--cities', required=False, default='train', help='Subset of MSLS')
        self.add_argument('--batch_size', type=int, default=16, help='input batch size')
        self.add_argument('--name', type=str, required=False, default='testexp', help='name of the experiment')
        self.add_argument('--backbone', type=str, default='vgg16', help='which architecture to use. [resnet50, resnet152, resnext, vgg16]')
        self.add_argument('--snapshot_dir', type=str, default='./snapshots', help='models are saved here')
        self.add_argument('--result_dir', type=str, default='./results', help='predictions and results are saved here')
        self.add_argument('--save_freq', type=int, default=1, help='save frequency in steps')
        self.add_argument('--dataset', type=str, default='soft_MSLS', help='[binary_MSLS|soft_MSLS]')
        self.add_argument('--pool', type=str, help='Global pool layer  max|avg|GeM')
        self.add_argument('--p', required=False, type=int, default=3, help='P parameter for GeM pool')
        self.add_argument('--norm', type=str, default="L2", help='Norm layer')
        self.add_argument('--image_size', type=str, default="480,640", help='Input size, separated by commas')
        self.add_argument('--last_layer', type=int, default=None, help='Last layer to keep')
        self.add_argument('--display_freq', type=int, default=10, help='frequency of showing training results on screen')
        self.add_argument('--steps', type=int, default=52, help='Number of training steps. 52= 1epoch')
        self.add_argument('--margin', type=float, default='.5', help='margin parameter for the contrastive loss')
        self.add_argument('--learning_rate', type=float, default='.1', help='learning rate')
        self.add_argument('--lr_gamma', type=float, default='.1', help='learning rate decay')
        self.add_argument('--step_size', type=float, default='25', help='Learning rate update frequency (in steps)')
        self.add_argument('--use_cpu', dest='use_gpu', help='Use CPU mode', action='store_false')
        self.set_defaults(use_gpu=True)


def extract_features(dataloader, network, feature_length, features_file):
    feats = np.zeros((len(dataloader.dataset), feature_length))
    for i, batch in tqdm(enumerate(dataloader), desc="Extracting features"):
        x = network.forward_single(batch.cuda())
        feats[i * dataloader.batch_size:i * dataloader.batch_size + dataloader.batch_size] = x.cpu().detach().squeeze(0)
    np.save(features_file, feats)


def validate(root_dir, result_dir, snapshot_dir, model, transformer, best_metric, reference_metric="recall@5"):
    print("Validating...")
    val_cities = ["cph", "sf"]
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(snapshot_dir, exist_ok=True)
    is_best = False
    model.eval()
    name = params.name
    retrieved_file = os.path.join(result_dir, f"{name}_retrieved.csv")

    for city in val_cities:
        print(city)

        query_index_file = os.path.join(root_dir, "train_val", city, "query.json")
        query_dataloader = create_dataloader("test", root_dir, query_index_file, None, transformer, 2)

        map_index_file = os.path.join(root_dir, "train_val", city, "database.json")
        map_dataloader = create_dataloader("test", root_dir, map_index_file, None, transformer, 2)

        query_features_file = os.path.join(result_dir, f"{name}_{city}_query_features.npy")
        extract_features(query_dataloader, model, model.feature_length, query_features_file)

        map_features_file = os.path.join(result_dir, f"{name}_{city}_database_features.npy")
        extract_features(map_dataloader, model, model.feature_length, map_features_file)

        extract_msls_top_k(map_features_file, query_features_file, map_index_file, query_index_file, retrieved_file, 25)

    results_file = os.path.join(result_dir, f"{name}_val_results.txt")
    metrics = msls_validate(retrieved_file, root_dir, results_file)

    if metrics[reference_metric] > best_metric:
        shutil.copy(retrieved_file, retrieved_file.replace(".csv", "_best.csv"))
        shutil.copy(results_file, results_file.replace(".txt", "_best.txt"))
        for city in val_cities:
            query_features_file = os.path.join(result_dir, f"{name}_{city}_query_features.npy")
            shutil.copy(query_features_file, query_features_file.replace(".npy", "_best.npy"))
            map_features_file = os.path.join(result_dir, f"{name}_{city}_database_features.npy")
            shutil.copy(map_features_file, map_features_file.replace(".npy", "_best.npy"))
        is_best = True
    model.train()
    return metrics, is_best


def image_transformer(image_size):
    if image_size[0] == image_size[1]:  # If we want to resize to square, we do resize+crop
        return ttf.Compose([
            ttf.Resize(size=(image_size[0])),
            ttf.CenterCrop(size=(image_size[0])),
            ttf.ToTensor(),
            ttf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return ttf.Compose([
            ttf.Resize(size=(image_size[0], image_size[1])),
            ttf.ToTensor(),
            ttf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def train(params):
    image_size = [int(x) for x in params.image_size.split(",")]
    print("training with images of size", image_size[0], image_size[1])

    transformer = image_transformer(image_size)
    model = create_model(params.backbone, params.pool, last_layer=params.last_layer, norm=params.norm, p_gem=params.p)
    loss = ContrastiveLoss(params.margin)
    optimizer = optim.SGD(model.parameters(), lr=params.learning_rate, weight_decay=0)
    optimizer.zero_grad()
    scheduler = StepLR(optimizer, step_size=params.step_size, gamma=params.lr_gamma)

    if params.use_gpu and torch.cuda.is_available():
        model = model.cuda()
        loss = loss.cuda()

    print(params.dataset)
    dataloader = create_msls_dataloader(
        params.dataset,
        params.root_dir,
        params.cities,
        transform=transformer,
        batch_size=params.batch_size
    )

    init_step = 0
    best_metric = 0
    total_iterations = 0
    best_metrics = 0
    ref_metric = "recall@5"
    error: torch.Tensor = None
    null_losses: torch.Tensor = None

    for step in tqdm(range(init_step, params.steps), desc="Steps"):
        e_iteration = 0
        for i, data in enumerate(dataloader):
            e_iteration += params.batch_size
            mini_batch_size = 2
            accum_iterations = int(data["im0"].shape[0] / mini_batch_size)
            for j in range(accum_iterations):
                a = j * mini_batch_size
                b = a + mini_batch_size

                if params.use_gpu:
                    x0, x1 = model(data["im0"][a:b, :].cuda(), data["im1"][a:b, :].cuda())
                    error = loss(x0, x1, (data["label"][a:b]).cuda())
                else:
                    x0, x1 = model(data["im0"][a:b, :], data["im1"][a:b, :])
                    error = loss(x0, x1, data["label"][a:b])

                null_losses = torch.sum(error == 0).item() / len(error)
                error = torch.mean(error) / accum_iterations
                error.backward()
                total_iterations += mini_batch_size

            if i % params.display_freq == 0:
                print(f"Step {step}, Iteration {e_iteration}, Loss {error:.4f}, Null loss {null_losses:.4f}")
            optimizer.step()
            optimizer.zero_grad()

        metrics, is_best = validate(params.root_dir, params.result_dir, params.snapshot_dir, model,
                                    transformer, best_metric, ref_metric)

        if step % params.save_freq == 0:
            save_path = os.path.join(params.snapshot_dir, f"{params.name}.pth")
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, save_path)

        if is_best:
            best_metric = metrics[ref_metric]
            best_metrics = metrics
            save_path = os.path.join(params.snapshot_dir, f"{params.name}_best.pth")
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, save_path)

        scheduler.step()
        dataloader.dataset.load_pairs()

    print("Done. Best results on val:")
    print(best_metrics)


if __name__ == "__main__":
    parser = TrainParser()
    params = parser.parse_args()
    train(params)
