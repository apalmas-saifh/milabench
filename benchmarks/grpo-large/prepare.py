#!/usr/bin/env python

from dataclasses import dataclass

from argklass import ArgumentParser
from benchmate.hugginface import download_hf_dataset, download_hf_model


@dataclass
class Arguments:
    dataset_name: str = "nvidia/AceReason-Math"
    dataset_train_split: str = "train"
    dataset_config: str = None
    model_name_or_path: str = "Qwen/Qwen2.5-72B"


def arguments() -> Arguments:
    parser = ArgumentParser()
    parser.add_arguments(Arguments)
    args, _ = parser.parse_known_args()
    return args


def new_prepare():
    args = arguments()

    download_hf_dataset(args.dataset_name, args.dataset_train_split, name=args.dataset_config)
    download_hf_model(args.model_name_or_path)

    print("=" * 60)
    print("Prepare script completed successfully")
    print("=" * 60)


if __name__ == "__main__":
    new_prepare()
