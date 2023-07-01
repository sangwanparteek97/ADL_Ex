import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
from pathlib import Path
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import random
from ex02_model import Unet
from ex02_diffusion import Diffusion, linear_beta_schedule
from torchvision.utils import save_image

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Train a neural network to diffuse images')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--timesteps', type=int, default=100, help='number of timesteps for diffusion model (default: 100)')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.003, help='learning rate (default: 0.003)')
    # parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    # parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=100, help='how many batches to wait before logging training status')
    parser.add_argument('--save_model', action='store_true', default=False, help='For Saving the current Model')
    parser.add_argument('--run_name', type=str, default="DDPM")
    parser.add_argument('--dry_run', action='store_true', default=False, help='quickly check a single pass')
    return parser.parse_args()


def sample_and_save_images(n_images, diffusor, model, device, args,store_path,num_classes):
    # TODO: Implement - adapt code and method signature as needed
    model.eval()
    #Images = diffusor.sample(model, 32, batch_size=n_images, channels=3)
    if args.run_name=="classifier_free_guidance":
        w=7
        image_classes=torch.randint(0, num_classes, (n_images,)).cuda()
        Images = diffusor.sample(model=model,image_size=32,batch_size=n_images,classes=image_classes,w=w)
    else:
        Images = diffusor.sample(model=model,image_size=32,batch_size=n_images)

    # for i, img in enumerate(Images):
    #     img = (img + 1) / 2  # Normalize image
    #     img_tensor = torch.from_numpy(img)
    #     save_path = f"{store_path}/image_{i}.png"  # Modify the path and file name as needed
    #     save_image(img_tensor, save_path)
    #     #save_image(img_tensor, store_path, nrow=int(np.sqrt(n_images)))
    for i,img in enumerate(Images):
        img_tensor = torch.from_numpy(img)
        filename="image{}.png".format(i)
        save_image(img_tensor,store_path+filename)


def test(model, testloader, diffusor, device, args):
    # TODO: Implement - adapt code and method signature as needed
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            t = torch.randint(0, diffusor.timesteps, (len(images),), device=device).long()
            if args.run_name=="classifier_free_guidance":
                loss =  diffusor.p_losses(model, images, t, loss_type="l2",classes=labels).item()
            else:
                loss = diffusor.p_losses(model, images,t, loss_type="l2").item()
            test_loss += loss
    test_loss /= len(testloader)
    print('Test Loss: {:.6f}'.format(test_loss))


def train(model, trainloader, optimizer, diffusor, epoch, device, args):
    batch_size = args.batch_size
    timesteps = args.timesteps

    pbar = tqdm(trainloader)
    for step, (images, labels) in enumerate(pbar):

        images = images.to(device)
        optimizer.zero_grad()

        # Algorithm 1 line 3: sample t uniformly for every example in the batch
        t = torch.randint(0, timesteps, (len(images),), device=device).long()
        if args.run_name=="classifier_free_guidance":
            loss = diffusor.p_losses(model, images, t, loss_type="l2",classes=labels)
        else:
            loss = diffusor.p_losses(model, images, t, loss_type="l2")

        loss.backward()
        optimizer.step()

        if step % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, step * len(images), len(trainloader.dataset),
                100. * step / len(trainloader), loss.item()))
        if args.dry_run:
            break


# def test(args):
#     # TODO (2.2): implement testing functionality, including generation of stored images.
#     pass


def run(args):
    timesteps = args.timesteps
    image_size = 32  # TODO (2.5): Adapt to new dataset
    channels = 3
    epochs = args.epochs
    batch_size = args.batch_size
    num_classes = 10
    device = "cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu"
    if args.run_name=="classifier_free_guidance":
        model = Unet(dim=image_size, channels=channels, dim_mults=(1, 2, 4,),class_free_guidance=True,p_uncond=0.2,num_classes=num_classes).to(device)
    else:
        model = Unet(dim=image_size, channels=channels, dim_mults=(1, 2, 4,)).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    my_scheduler = lambda x: linear_beta_schedule(0.0001, 0.02, x)
    diffusor = Diffusion(timesteps, my_scheduler, image_size, device)

    # define image transformations (e.g. using torchvision)
    transform = Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),    # turn into torch Tensor of shape CHW, divide by 255
        transforms.Lambda(lambda t: (t * 2) - 1)   # scale data to [-1, 1] to aid diffusion process
    ])
    reverse_transform = Compose([
        Lambda(lambda t: (t.clamp(-1, 1) + 1) / 2),
        Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        Lambda(lambda t: t * 255.),
        Lambda(lambda t: t.numpy().astype(np.uint8)),
        ToPILImage(),
    ])

    dataset = datasets.CIFAR10('/proj/aimi-adl/CIFAR10/', download=True, train=True, transform=transform)
    trainset, valset = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.9), len(dataset) - int(len(dataset) * 0.9)])
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)

    # Download and load the test data
    testset = datasets.CIFAR10('/proj/aimi-adl/CIFAR10/', download=True, train=False, transform=transform)
    testloader = DataLoader(testset, batch_size=int(batch_size/2), shuffle=True)

    for epoch in range(epochs):
        train(model, trainloader, optimizer, diffusor, epoch, device, args)
        test(model, valloader, diffusor, device, args)

    test(model, testloader, diffusor, device, args)

    save_path = "/home/cip/ai2022/wu58gudu/ADL_Ex/ex_02/Sample_Images/"  # TODO: Adapt to your needs
    n_images = 8
    sample_and_save_images(n_images, diffusor, model, device, save_path,num_classes=10)
    torch.save(model.state_dict(), os.path.join("/home/cip/ai2022/wu58gudu/ADL_Ex/ex_02/models/", args.run_name, f"ckpt.pt"))


if __name__ == '__main__':
    args = parse_args()
    # TODO (2.2): Add visualization capabilities
    run(args)
    save_path = "/home/cip/ai2022/wu58gudu/ADL_Ex/ex_02/"
    # image_files = [f for f in os.listdir(save_path) if os.path.isfile(os.path.join(save_path, f))]
    # image_tensors = []
    image_dir = '/home/cip/ai2022/wu58gudu/ADL_Ex/ex_02/'

    # Get the list of image files
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

    # Select 10 random image files
    selected_image_files = random.sample(image_files, 10)

    # Create a figure with a grid of subplots
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))

    # Iterate over the selected image files and plot them
    for i, image_file in enumerate(selected_image_files):
        # Read the image file
        image_path = os.path.join(image_dir, image_file)
        image = plt.imread(image_path)

        # Get the corresponding subplot axes
        ax = axes[i // 5, i % 5]

        # Plot the image
        ax.imshow(image)
        ax.axis('off')

    # Adjust the spacing between subplots
    plt.subplots_adjust(hspace=0.3, wspace=0.05)

    # Show the plot
    plt.show()
