DATASET_PATHS = {
    "celeba_train": "datasets/ffhq",
    "celeba_test": "datasets/ffhq",
    "celeba_train_sketch": "datasets/celeba/img_align_celeba",
    "celeba_test_sketch": "datasets/celeba/img_align_celeba",
    "celeba_train_segmentation": "datasets/celeba/img_align_celeba",
    "celeba_test_segmentation": "datasets/celeba/img_align_celeba",
    "ffhq": "../datasets/ffhq",
}

MODEL_PATHS = {
    "stylegan_ffhq": "pretrained_models/stylegan2-ffhq-config-f.pt",
    "ir_se50": "pretrained_models/model_ir_se50.pth",
    "circular_face": "pretrained_models/CurricularFace_Backbone.pth",
    "mtcnn_pnet": "pretrained_models/mtcnn/pnet.npy",
    "mtcnn_rnet": "pretrained_models/mtcnn/rnet.npy",
    "mtcnn_onet": "pretrained_models/mtcnn/onet.npy",
    "shape_predictor": "shape_predictor_68_face_landmarks.dat",
}
