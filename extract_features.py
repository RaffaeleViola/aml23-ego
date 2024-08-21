from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.models import AlexNet_Weights
import torch.nn.functional as F
from torchmetrics import Accuracy
from tqdm import tqdm

from utils.loaders import EpicKitchensDataset

#### DATA SETUP
# Define the transforms to use on images
# dataset_transform = T.Compose([
#     T.Resize(256),
#     T.CenterCrop(224),
#     T.ToTensor(),
#     T.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# Define the Dataset object for training & testing
# train_dataset = PACSDataset(domain='cartoon', transform=dataset_transform)
# test_dataset = PACSDataset(domain='sketch', transform=dataset_transform)
test_dataset = EpicKitchensDataset("train_val\D1_test.pkl",)
# Define the DataLoaders
# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=4)


#### ARCHITECTURE SETUP
# Create the Network Architecture object
model = AlexNet()
# Load pre-trained weights
model.load_state_dict(AlexNet_Weights.IMAGENET1K_V1.get_state_dict(progress=True), strict=False)
# Overwrite the final classifier layer as we only have 7 classes in PACS
model.classifier[-1] = nn.Linear(4096, NUM_CLASSES)


#### TRAINING SETUP
# Move model to device before passing it to the optimizer
model = model.to(DEVICE)

# Create Optimizer & Scheduler objects
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)


#### TRAINING LOOP
model.train()
if False:
    # Baseline
    for epoch in range(NUM_EPOCHS):
        epoch_loss = [0.0, 0]
        for x, y in tqdm(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            # x --> [B x C x H x W]
    
            # Category Loss
            cls_o, _ = model(x)
            loss = F.cross_entropy(cls_o, y)
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            epoch_loss[0] += loss.item()
            epoch_loss[1] += x.size(0)
            
        scheduler.step()
        print(f'[EPOCH {epoch+1}] Avg. Loss: {epoch_loss[0] / epoch_loss[1]}')
else:
    # DANN
    LAMBDA = 1e-4
    for epoch in range(NUM_EPOCHS):
        epoch_loss = [0.0, 0]
        for batch_idx, ((src_x, src_y), (trg_x, _)) in tqdm(enumerate(zip(train_loader, test_loader))):
            src_x, src_y = src_x.to(DEVICE), src_y.to(DEVICE)
            trg_x = trg_x.to(DEVICE)

            src_cls_o, src_dom_o = model(src_x)
            _, trg_dom_o = model(trg_x)

            if batch_idx % 2 == 0:
                # Classification Loss
                loss = F.cross_entropy(src_cls_o, src_y)
                
            else:
                # Classification Loss
                cls_loss = F.cross_entropy(src_cls_o, src_y)
    
                # Source Domain Adversarial Loss --> src_dom_label = 0
                src_dom_label = torch.zeros(src_dom_o.size(0)).long().to(DEVICE)
                src_dom_loss = F.cross_entropy(src_dom_o, src_dom_label)
    
                # Target Domain Adversarial Loss --> trg_dom_label = 1
                trg_dom_label = torch.ones(trg_dom_o.size(0)).long().to(DEVICE)
                trg_dom_loss = F.cross_entropy(trg_dom_o, trg_dom_label)
    
                # Final Loss
                loss = cls_loss - LAMBDA * (src_dom_loss + trg_dom_loss)
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            epoch_loss[0] += loss.item()
            epoch_loss[1] += src_x.size(0)
            
        scheduler.step()
        print(f'[EPOCH {epoch+1}] Avg. Loss: {epoch_loss[0] / epoch_loss[1]}')
# pip install wandb

#### TEST LOOP
model.eval()

meter = Accuracy(task='multiclass', num_classes=NUM_CLASSES).to(DEVICE)

with torch.no_grad():
    for x, y in tqdm(test_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        cls_o, _ = model(x)
        meter.update(cls_o, y)
accuracy = meter.compute()

print(f'\nAccuracy on the target domain: {100 * accuracy:.2f}%')




# # load the pretrained model
# # set the extraction only features of the step before the last7
# # model set in evaluation mode
# # load the dataset ( train probably )
# # extract the wanted features
# # save features in a .pth or .kpl file
# import pickle
# from utils.logger import logger, get_handler
# import torch
# from utils.loaders import EpicKitchensDataset, ActionNetDataset
# from utils.args import init_args
# from utils.utils import pformat_dict
# import utils
# import numpy as np
# import os
# import models as model_list
# import tasks
# from utils.torch_device import get_device

# # Global variables among training functions
# modalities = None
# np.random.seed(13696641)
# torch.manual_seed(13696641)
# args = None

# def init_operations():
#     global args
#     args = init_args()
#     logger.addHandler(get_handler(args.logfile))
#     logger.info("Feature Extraction")
#     logger.info("Running with parameters: " + pformat_dict(args, indent=1))

#     if args.gpus is not None:
#         logger.debug("Using only these GPUs: {}".format(args.gpus))
#         os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

# def main():
#     global modalities
#     init_operations()
#     modalities = args.modality

#     # Recover valid paths, domains, classes
#     num_classes, valid_labels, source_domain, target_domain = utils.utils.get_domains_and_labels(args)
#     device = torch.device(get_device())
#     logger.info("Device: {}".format(device))

#     models = {}
#     logger.info("Instantiating models per modality")
#     for m in modalities:
#         logger.info("{} Net\tModality: {}".format(args.models[m].model, m))
#         models[m] = getattr(model_list, args.models[m].model)(num_classes, m, args.models[m], **args.models[m].kwargs)

#     action_classifier = tasks.ActionRecognition(
#         "action-classifier", models, 1, args.total_batch, args.models_dir, num_classes, args.save.num_clips, args.models, args=args
#     )
#     action_classifier.load_on_gpu(device)
#     if args.resume_from is not None:
#         action_classifier.load_last_model(args.resume_from)

#     if args.action == "save":
#         for dataset_name in ["epic_kitchens", "action_net"]:
#             if dataset_name == "epic_kitchens":
#                 loader = torch.utils.data.DataLoader(
#                     EpicKitchensDataset(
#                         args.dataset.shift.split("-")[1],
#                         modalities,
#                         args.split,
#                         args.dataset,
#                         args.save.num_frames_per_clip,
#                         args.save.num_clips,
#                         args.save.dense_sampling,
#                         None,  # No augmentation for feature extraction
#                         additional_info=True,
#                         **{"save": args.split}
#                     ),
#                     batch_size=1,
#                     shuffle=False,
#                     num_workers=args.dataset.workers,
#                     pin_memory=True,
#                     drop_last=False
#                 )
#             else:
#                 loader = torch.utils.data.DataLoader(
#                     ActionNetDataset(
#                         args.dataset.shift.split("-")[1],
#                         modalities,
#                         args.split,
#                         args.dataset,
#                         args.save.num_frames_per_clip,
#                         args.save.num_clips,
#                         args.save.dense_sampling,
#                         None,  # No augmentation for feature extraction
#                         additional_info=True,
#                         **{"save": args.split}
#                     ),
#                     batch_size=1,
#                     shuffle=False,
#                     num_workers=args.dataset.workers,
#                     pin_memory=True,
#                     drop_last=False
#                 )
            
#             save_feat(action_classifier, loader, device, action_classifier.current_iter, num_classes, dataset_name)
#     else:
#         raise NotImplementedError

# def save_feat(model, loader, device, it, num_classes, dataset_name):
#     global modalities

#     model.reset_acc()
#     model.train(False)
#     results_dict = {"features": []}
#     num_samples = 0
#     logits = {}
#     features = {}

#     with torch.no_grad():
#         for i_val, (data, label, video_name, uid) in enumerate(loader):
#             label = label.to(device)

#             for m in modalities:
#                 batch, _, height, width = data[m].shape
#                 data[m] = data[m].reshape(
#                     batch,
#                     args.save.num_clips,
#                     args.save.num_frames_per_clip[m],
#                     -1,
#                     height,
#                     width,
#                 )
#                 data[m] = data[m].permute(1, 0, 3, 2, 4, 5)

#                 logits[m] = torch.zeros((args.save.num_clips, batch, num_classes)).to(
#                     device
#                 )
#                 features[m] = torch.zeros(
#                     (args.save.num_clips, batch, model.task_models[m].module.feat_dim)
#                 ).to(device)

#             clip = {}
#             for i_c in range(args.save.num_clips):
#                 for m in modalities:
#                     clip[m] = data[m][i_c].to(device)

#                 output, feat = model(clip)
#                 feat = feat["features"]
#                 for m in modalities:
#                     logits[m][i_c] = output[m]
#                     features[m][i_c] = feat[m]
#             for m in modalities:
#                 logits[m] = torch.mean(logits[m], dim=0)
#             for i in range(batch):
#                 sample = {
#                     "uid": int(uid[i].cpu().detach().numpy()),
#                     "video_name": video_name[i],
#                 }
#                 for m in modalities:
#                     sample["features_" + m] = features[m][:, i].cpu().detach().numpy()
#                 results_dict["features"].append(sample)
#             num_samples += batch

#             model.compute_accuracy(logits, label)

#             if (i_val + 1) % (len(loader) // 5) == 0:
#                 logger.info(
#                     "[{}/{}] top1= {:.3f}% top5 = {:.3f}%".format(
#                         i_val + 1,
#                         len(loader),
#                         model.accuracy.avg[1],
#                         model.accuracy.avg[5],
#                     )
#                 )

#         os.makedirs("saved_features", exist_ok=True)
#         pickle.dump(
#             results_dict,
#             open(
#                 os.path.join(
#                     "saved_features",
#                     args.name
#                     + "_"
#                     + dataset_name
#                     + "_"
#                     + args.split
#                     + ".pkl",
#                 ),
#                 "wb",
#             ),
#         )

# if __name__ == "__main__":
#     main()

# # import subprocess
# # from multiprocessing import Pool, Process
# # from utils.logger import logger

# # FRAMES = [5, 10, 25]
# # SPLITS = ["test", "train"]
# # SAMPLING = [True, False]


# # def run_command(args):
# #     command = [
# #         "python3",
# #         "save_feat.py",
# #         f"name={args['name']}",
# #         f"config={args['config']}",
# #         f"dataset.shift={args['shift']}",
# #         f"dataset.RGB.data_path={args['data_path']}",
# #         f"split={args['split']}",
# #         f"save.num_frames_per_clip.RGB={args['num_frames_per_clip']}",
# #         f"save.dense_sampling.RGB={args['sampling']}",
# #     ]
# #     subprocess.run(command)


# # # Maximum number of parallel executions
# # max_processes = 2  # Adjust as needed


# # if __name__ == "__main__":
# #     logger.info("Starting feature extraction")
# #     arguments_list = []
# #     for i in FRAMES:
# #         for j in SPLITS:
# #             for k in SAMPLING:
# #                 arguments_list.append(
# #                     {
# #                         "name": f"feat_{i}_{j}_{k}",
# #                         "config": "configs/I3D_save_feat.yaml",
# #                         "shift": "D1-D1",
# #                         "data_path": "data/EK",
# #                         "split": j,
# #                         "num_frames_per_clip": i,
# #                         "sampling": k,
# #                     }
# #                 )

# #     # Execute commands sequentially
# #     for args in arguments_list:
# #         run_command(args)
