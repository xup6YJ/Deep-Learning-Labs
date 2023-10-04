import os
import random
import numpy as np
import argparse
from tqdm.auto import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from accelerate import Accelerator
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMScheduler
from model import MyConditionedUNet
from dataset import ICLEVR_Dataset

from utils import evaluate, plot_result
from evaluator import evaluation_model

if __name__ == "__main__":
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default="../data/iclevr", type=str)
    parser.add_argument('--log_dir', default="./logs", type=str)
    parser.add_argument('--figure_dir', default="figures", type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--num_epochs', default=150, type=int) # default 150
    parser.add_argument('--sample_size', default=64, type=int)
    parser.add_argument('--beta_schedule', choices=['linear', 'cosine'], default='cosine', type=str)
    parser.add_argument('--predict_type', default="epsilon", type=str)
    parser.add_argument('--block_dim', default=128, type=int)
    parser.add_argument('--layers_per_block', default=2, type=int)
    parser.add_argument('--embed_type', default="timestep", type=str)
    parser.add_argument('--lr_warmup_steps', default=500, type=int)
    parser.add_argument('--mixed_precision', default="fp16", type=str)
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--resblock', action='store_true' ,default=False)
    parser.add_argument('--label_emb', action='store_true' ,default=False)

    args = parser.parse_args()

    # set reproducibility
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("> Using device: {}".format(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"))

    format_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    name = 'created_time={}-lr={:.5f}-lr_warmup={}-block_dim={}-layers={}-embed={}-schedule={}-predict_type={}'.format(format_time, args.lr, args.lr_warmup_steps, args.block_dim, args.layers_per_block, args.embed_type, args.beta_schedule, args.predict_type)
    args.log_dir = './%s/%s' % (args.log_dir, name)
    args.figure_dir = '%s/%s' % (args.log_dir, args.figure_dir)

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.figure_dir, exist_ok=True)

    # 覆寫舊的 train_record.txt，重新記錄 log
    with open('{}/train_record.txt'.format(args.log_dir), 'w') as train_record:
        train_record.write('args: {}\n'.format(args))

    sample_size = args.sample_size
    block_dim = args.block_dim
    layers = args.layers_per_block
    embed_type = args.embed_type
    num_epochs = args.num_epochs
    lr = args.lr
    batch_size = args.batch_size

    if args.resblock:
        model = MyConditionedUNet(
            args=args,
            sample_size=sample_size,       # the target image resolution
            in_channels=3,                 # additional input channels for class condition
            out_channels=3,
            layers_per_block=layers,
            block_out_channels=(block_dim, block_dim, block_dim*2, block_dim*2, block_dim*4, block_dim*4),
            down_block_types=(
                "ResnetDownsampleBlock2D",          # a regular ResNet downsampling block
                "ResnetDownsampleBlock2D",
                "ResnetDownsampleBlock2D",
                "ResnetDownsampleBlock2D",
                "AttnDownBlock2D",      # a ResNet downsampling block with spatial self-attention
                "ResnetDownsampleBlock2D",
            ),
            up_block_types=(
                "ResnetUpsampleBlock2D",
                "AttnUpBlock2D",        # a ResNet upsampling block with spatial self-attention
                "ResnetUpsampleBlock2D",            # a regular ResNet upsampling block
                "ResnetUpsampleBlock2D",
                "ResnetUpsampleBlock2D",
                "ResnetUpsampleBlock2D",
            ),
            class_embed_type=embed_type,
        )


    else:
        model = MyConditionedUNet(
            args=args,
            sample_size=sample_size,       # the target image resolution
            in_channels=3,                 # additional input channels for class condition
            out_channels=3,
            layers_per_block=layers,
            block_out_channels=(block_dim, block_dim, block_dim*2, block_dim*2, block_dim*4, block_dim*4),
            down_block_types=(
                "DownBlock2D",          # a regular ResNet downsampling block
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",      # a ResNet downsampling block with spatial self-attention
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",        # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",            # a regular ResNet upsampling block
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
            class_embed_type=embed_type,
        )

    beta_schedule = "squaredcos_cap_v2" if args.beta_schedule == "cosine" else "linear"
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule=beta_schedule)

    transform = transforms.Compose([
        transforms.Resize((sample_size, sample_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    dataset = ICLEVR_Dataset(args, mode='train', transforms=transform)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=len(train_loader) * num_epochs,
    )
    evaluation = evaluation_model()

    state = {
        "model": model.state_dict(),
        "best_epoch": 0,
        "test_acc": 0,
        "new_test_acc": 0,
    }

    losses = []
    acc_list = []
    new_acc_list = []
    best_acc = 0
    best_new_acc = 0
    global_step = 0

    # Accelerator
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(args.log_dir, "logging"),
    )

    if accelerator.is_main_process:
        accelerator.init_trackers("train_example")

    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, lr_scheduler
    )

    # start training
    for epoch in range(num_epochs):
        progress_bar = tqdm(total=len(train_loader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch+1}/{num_epochs}")

        total_loss = 0
        for i, (x, class_label) in enumerate(train_loader):
            x, class_label = x.to(device), class_label.to(device)

            # sample some noise
            noise = torch.randn_like(x)

            # sample random timesteps
            timesteps = torch.randint(0, 1000, (x.shape[0],)).long().to(device)

            # add noise to the image and get the noisy image
            noisy_image = noise_scheduler.add_noise(x, noise, timesteps)

            with accelerator.accumulate(model):
                # get the model prediction
                noise_pred = model(noisy_image, timesteps, class_label).sample

                # calculate the loss
                loss = loss_fn(noise_pred, noise)
                total_loss += loss.item()
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            logs = {"loss": total_loss / (i+1), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.update(1)
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        losses.append(total_loss / len(train_loader))

        # testing
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            print('> Testing on "test"...')
            test_image, test_label = evaluate(model, noise_scheduler, args, device, "test")
            test_acc = evaluation.eval(test_image, test_label)

            print('> Testing on "new_test"...')
            new_test_image, new_test_label = evaluate(model, noise_scheduler, args, device, "new_test")
            new_test_acc = evaluation.eval(new_test_image, new_test_label)

            image = (test_image / 2 + 0.5).clamp(0, 1)
            save_image(make_grid(image, nrow=8), f"{args.figure_dir}/[epoch:{epoch}]_Test_[acc:{test_acc:.4f}].png")

            image = (new_test_image / 2 + 0.5).clamp(0, 1)
            save_image(make_grid(image, nrow=8), f"{args.figure_dir}/[epoch:{epoch}]_NewTest_[acc:{new_test_acc:.4f}].png")

            acc_list.append(test_acc)
            new_acc_list.append(new_test_acc)
            print("> Accuracy: [Test]: {:.4f}, [New Test]: {:.4f}".format(test_acc, new_test_acc))

            with open('{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
                train_record.write(('[Epoch: %02d] loss: %.5f | test acc: %.5f | new_test acc: %.5f\n' % (epoch, losses[-1], test_acc, new_test_acc)))

            if test_acc >= best_acc or new_test_acc >= best_new_acc:
                state["model"] = model.state_dict()
                state["best_epoch"] = epoch
                state["test_acc"] = test_acc
                state["new_test_acc"] = new_test_acc
                best_acc = test_acc
                best_new_acc = new_test_acc

                formatted_time = datetime.now().strftime('%Y%m%d_%H%M%S')

                torch.save(state, os.path.join(args.log_dir, f"{formatted_time}_[epoch:{epoch}]_[test_acc:{test_acc:.4f}]_[new_test_acc:{new_test_acc:.4f}].pth"))
                print("> New checkpoint")

    plot_result(losses, acc_list, new_acc_list, args)