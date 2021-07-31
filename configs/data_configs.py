from configs import transforms_config
from configs.paths_config import DATASET_PATHS


DATASETS = {
    "ffhq_encode": {
        "transforms": transforms_config.EncodeTransforms,
        "train_source_root": DATASET_PATHS["ffhq"],
        "train_target_root": DATASET_PATHS["ffhq"],
        "test_source_root": DATASET_PATHS["celeba_test"],
        "test_target_root": DATASET_PATHS["celeba_test"],
    },
    "ffhq_frontalize": {
        "transforms": transforms_config.FrontalizationTransforms,
        "train_source_root": DATASET_PATHS["ffhq"],
        "train_target_root": DATASET_PATHS["ffhq"],
        "test_source_root": DATASET_PATHS["celeba_test"],
        "test_target_root": DATASET_PATHS["celeba_test"],
    },
    "celebs_sketch_to_face": {
        "transforms": transforms_config.SketchToImageTransforms,
        "train_source_root": DATASET_PATHS["celeba_train_sketch"],
        "train_target_root": DATASET_PATHS["celeba_train"],
        "test_source_root": DATASET_PATHS["celeba_test_sketch"],
        "test_target_root": DATASET_PATHS["celeba_test"],
    },
    "celebs_seg_to_face": {
        "transforms": transforms_config.SegToImageTransforms,
        "train_source_root": DATASET_PATHS["celeba_train_segmentation"],
        "train_target_root": DATASET_PATHS["celeba_train"],
        "test_source_root": DATASET_PATHS["celeba_test_segmentation"],
        "test_target_root": DATASET_PATHS["celeba_test"],
    },
    "celebs_super_resolution": {
        "transforms": transforms_config.SuperResTransforms,
        "train_source_root": DATASET_PATHS["celeba_train"],
        "train_target_root": DATASET_PATHS["celeba_train"],
        "test_source_root": DATASET_PATHS["celeba_test"],
        "test_target_root": DATASET_PATHS["celeba_test"],
    },
    "celebs_noise_reduction": {
        "transforms": transforms_config.NoiseTransforms,
        "train_source_root": DATASET_PATHS["celeba_train"],
        "train_target_root": DATASET_PATHS["celeba_train"],
        "test_source_root": DATASET_PATHS["celeba_test"],
        "test_target_root": DATASET_PATHS["celeba_test"],
    },
    "special_dataset": {
        "transforms": transforms_config.SuperResTransforms,
        "train_source_root": "../images/train/",
        "train_target_root": "../images/train/",
        "test_source_root": "../images/train/",
        "test_target_root": "../images/train/",
    },
}
