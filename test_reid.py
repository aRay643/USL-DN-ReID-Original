import os
import random
import numpy as np
import argparse
import torch
import torch.nn as nn

from config import cfg
from datasets.make_dataloader import (
    make_dataset,
    make_dataset_dnwild,
    make_sysu_dataset_manager,
    make_testloader_sysu,
    make_testloader_usl_dnreid,
)
from pcl.model import make_model
from pcl.processor_pcl import do_inference, do_inference_sysu

def set_seed(seed):
    """Set random seeds to ensure reproducibility during inference."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_dataset_name(name):
    return str(name).strip().lower().replace("_", "-")


def evaluate_sysu(cfg, model, dataset):
    metric_keys = ("mAP", "mINP", "rank1", "rank5", "rank10", "rank20")
    results = {}
    for mode in ("all", "indoor"):
        trial_results = []
        for trial in range(10):
            print(
                "-------- SYSU-MM01 {}-search single-shot trial {}/10 --------".format(
                    mode, trial + 1
                )
            )
            val_loader, num_query = make_testloader_sysu(
                cfg, dataset, mode=mode, trial=trial
            )
            trial_result = do_inference_sysu(cfg, model, val_loader, num_query)
            trial_results.append(trial_result)
            print(
                "Trial {} -> Rank-1: {:.1%}, Rank-5: {:.1%}, Rank-10: {:.1%}, "
                "Rank-20: {:.1%}, mAP: {:.1%}, mINP: {:.1%}".format(
                    trial,
                    trial_result["rank1"],
                    trial_result["rank5"],
                    trial_result["rank10"],
                    trial_result["rank20"],
                    trial_result["mAP"],
                    trial_result["mINP"],
                )
            )

        mean_result = {
            key: float(np.mean([result[key] for result in trial_results]))
            for key in metric_keys
        }
        results[mode] = mean_result
        print(
            "{}-search 10-trial mean -> Rank-1: {:.1%}, Rank-5: {:.1%}, "
            "Rank-10: {:.1%}, Rank-20: {:.1%}, mAP: {:.1%}, mINP: {:.1%}\n".format(
                mode,
                mean_result["rank1"],
                mean_result["rank5"],
                mean_result["rank10"],
                mean_result["rank20"],
                mean_result["mAP"],
                mean_result["mINP"],
            )
        )
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="USL-DN-ReID Inference (Cross-Domain Test)")
    parser.add_argument(
        "--config_file", default="config/dn348-vit.yml", help="path to config file", type=str
    )
    parser.add_argument(
        "--weights", default="", help="path to the pretrained model weights (.pth)", type=str
    )
    parser.add_argument(
        "opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER
    )
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    # 1. Initialize configuration
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    set_seed(42)

    # 2. Build dataset
    print(f"========== Loading Dataset: {cfg.DATASETS.NAMES} ==========")
    dataset_name = normalize_dataset_name(cfg.DATASETS.NAMES)
    sysu_dataset = None
    if dataset_name == "dn348":
        train_day, train_night, Test_day, Test_night = make_dataset(cfg)
    elif dataset_name in {"sysu", "sysu-mm01"}:
        sysu_dataset = make_sysu_dataset_manager(cfg)
        train_visible = sysu_dataset.train_visible
        train_infrared = sysu_dataset.train_infrared
        train_day, train_night = train_visible, train_infrared
    else:
        train_day, train_night, Query_day, Query_night, Test_day, Test_night = make_dataset_dnwild(cfg)

    # 3. Initialize model
    # Note: num_classes and num_img_* are placeholders required for model instantiation 
    # but are not used during inference.
    print("========== Initializing Model ==========")
    model = make_model(cfg, num_classes=3000, num_img_day=len(train_day), num_img_night=len(train_night))

    # Dynamically load weights
    if args.weights:
        print(f"Loading weights from: {args.weights}")
        model.load_param(args.weights)
    else:
        print("Warning: No weights provided (--weights). Evaluating with random initialization.")

    model.cuda()
    model = nn.DataParallel(model)
    model.eval()  # Set model to evaluation mode

    # 4. Execute cross-domain evaluation
    print("\n========== Starting Cross-Domain Evaluation ==========")
    if dataset_name == "dn348":
        # --- DN-348 Evaluation ---
        print("-------- Day to Night (D2N) Test --------")
        val_loader_d2n = make_testloader_usl_dnreid(cfg, Test_day, Test_night)
        map_d2n, r1_d2n, r5_d2n = do_inference(cfg, model, val_loader_d2n, len(Test_day))
        print(f"D2N Results -> mAP: {map_d2n:.1%}, Rank-1: {r1_d2n:.1%}, Rank-5: {r5_d2n:.1%}\n")

        print("-------- Night to Day (N2D) Test --------")
        val_loader_n2d = make_testloader_usl_dnreid(cfg, Test_night, Test_day)
        map_n2d, r1_n2d, r5_n2d = do_inference(cfg, model, val_loader_n2d, len(Test_night))
        print(f"N2D Results -> mAP: {map_n2d:.1%}, Rank-1: {r1_n2d:.1%}, Rank-5: {r5_n2d:.1%}\n")

    elif dataset_name in {"sysu", "sysu-mm01"}:
        # Official SYSU-MM01 direction: infrared query -> visible gallery.
        evaluate_sysu(cfg, model, sysu_dataset)
    else:
        # --- DN-Wild Evaluation ---
        print("-------- Day to Night (D2N) Test --------")
        val_loader_d2n = make_testloader_usl_dnreid(cfg, Query_day, Test_night)
        map_d2n, r1_d2n, r5_d2n = do_inference(cfg, model, val_loader_d2n, len(Query_day))
        print(f"D2N Results -> mAP: {map_d2n:.1%}, Rank-1: {r1_d2n:.1%}, Rank-5: {r5_d2n:.1%}\n")

        print("-------- Night to Day (N2D) Test --------")
        val_loader_n2d = make_testloader_usl_dnreid(cfg, Query_night, Test_day)
        map_n2d, r1_n2d, r5_n2d = do_inference(cfg, model, val_loader_n2d, len(Query_night))
        print(f"N2D Results -> mAP: {map_n2d:.1%}, Rank-1: {r1_n2d:.1%}, Rank-5: {r5_n2d:.1%}\n")

    print("========== Evaluation Finished ==========")
