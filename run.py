import numpy as np
import time
import argparse

from tqdm import tqdm
from keras.datasets import mnist, cifar10
from keras.models import load_model, Model
from sa import fetch_dsa, fetch_lsa, get_sc
from utils import *

CLIP_MIN = -0.5
CLIP_MAX = 0.5

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", "-d", help="Dataset", type=str, default="mnist")
    parser.add_argument(
        "--lsa", "-lsa", help="Likelihood-based Surprise Adequacy", action="store_true"
    )
    parser.add_argument(
        "--dsa", "-dsa", help="Distance-based Surprise Adequacy", action="store_true"
    )
    parser.add_argument(
        "--target",
        "-target",
        help="Target input set (test or adversarial set)",
        type=str,
        default="fgsm",
    )
    parser.add_argument(
        "--save_path", "-save_path", help="Save path", type=str, default="./tmp/"
    )
    parser.add_argument(
        "--batch_size", "-batch_size", help="Batch size", type=int, default=128
    )
    parser.add_argument(
        "--var_threshold",
        "-var_threshold",
        help="Variance threshold",
        type=int,
        default=1e-5,
    )
    parser.add_argument(
        "--upper_bound", "-upper_bound", help="Upper bound", type=int, default=2000
    )
    parser.add_argument(
        "--n_bucket",
        "-n_bucket",
        help="The number of buckets for coverage",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--num_classes",
        "-num_classes",
        help="The number of classes",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--is_classification",
        "-is_classification",
        help="Is classification task",
        type=bool,
        default=True,
    )
    args = parser.parse_args()
    assert args.d in ["mnist", "cifar"], "Dataset should be either 'mnist' or 'cifar'"
    assert args.lsa ^ args.dsa, "Select either 'lsa' or 'dsa'"
    print(args)

    if args.d == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)

        # Load pre-trained model.
        model = load_model("./model/model_mnist.h5")
        model.summary()

        # You can select some layers you want to test.
        # layer_names = ["activation_1"]
        # layer_names = ["activation_2"]
        layer_names = ["activation_3"]

        # Load target set.
        x_target = np.load("./adv/adv_mnist_{}.npy".format(args.target))

    elif args.d == "cifar":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        model = load_model("./model/model_cifar.h5")
        model.summary()

        # layer_names = [
        #     layer.name
        #     for layer in model.layers
        #     if ("activation" in layer.name or "pool" in layer.name)
        #     and "activation_9" not in layer.name
        # ]
        layer_names = ["activation_6"]

        x_target = np.load("./adv/adv_cifar_{}.npy".format(args.target))

    x_train = x_train.astype("float32")
    x_train = (x_train / 255.0) - (1.0 - CLIP_MAX)
    x_test = x_test.astype("float32")
    x_test = (x_test / 255.0) - (1.0 - CLIP_MAX)

    if args.lsa:
        test_lsa = fetch_lsa(model, x_train, x_test, "test", layer_names, args)

        target_lsa = fetch_lsa(model, x_train, x_target, args.target, layer_names, args)
        target_cov = get_sc(
            np.amin(target_lsa), args.upper_bound, args.n_bucket, target_lsa
        )

        auc = compute_roc_auc(test_lsa, target_lsa)
        print(infog("ROC-AUC: " + str(auc * 100)))

    if args.dsa:
        test_dsa = fetch_dsa(model, x_train, x_test, "test", layer_names, args)

        target_dsa = fetch_dsa(model, x_train, x_target, args.target, layer_names, args)
        target_cov = get_sc(
            np.amin(target_dsa), args.upper_bound, args.n_bucket, target_dsa
        )

        auc = compute_roc_auc(test_dsa, target_dsa)
        print(infog("ROC-AUC: " + str(auc * 100)))

    print(infog("{} coverage: ".format(args.target) + str(target_cov)))
