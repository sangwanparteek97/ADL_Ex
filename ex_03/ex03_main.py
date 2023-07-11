#  Standard libraries
import os
import numpy as np
import tqdm
import pandas as pd
import argparse
from typing import Union, Dict
import random
#  Imports for plotting
import matplotlib.pyplot as plt
import seaborn as sns

#  Imports for data loading
from pathlib import Path

#  PyTorch & DL
import torch
import torch.utils.data as data
import torch.optim as optim
import torchmetrics
import torchvision
import tensorboard

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

# Deterministic operations on GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

#  Misc
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc

from ex03_data import get_datasets, TransformTensorDataset
from ex03_model import ShallowCNN
from ex03_ood import score_fn


def parse_args():
    parser = argparse.ArgumentParser(description='Configure training/inference/sampling for EBMs')
    parser.add_argument('--data_dir', type=str, default="/proj/aimi-adl/GLYPHS/",
                        help='path to directory with glyph image data')
    parser.add_argument('--ckpt_dir', type=str, default="/home/cip/ai2022/qi27ycyt/models/",
                        help='path to directory where model checkpoints are stored')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--num_epochs', type=int, default=120,
                        help='number of epochs to train (default: 120)')
    parser.add_argument('--cbuffer_size', type=int, default=128,
                        help='num. images per class in the sampling reservoir (default: 128)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--lr_gamma', type=float, default=0.97,
                        help='exponentional learning rate decay factor (default: 0.97)')
    parser.add_argument('--lr_stepsize', type=int, default=2,
                        help='learning rate decay step size (default: 2)')
    parser.add_argument('--alpha', type=int, default=0.1,
                        help='strength of L2 regularization (default: 0.1)')
    parser.add_argument('--num_classes', type=int, default=42,
                        help='number of output nodes/classes (default: 1 (EBM), 42 (JEM))')
    parser.add_argument('--ccond_sample', type=bool, default=False,
                        help='flag that specifies class-conditional or unconditional sampling (default: false')
    parser.add_argument('--num_workers', type=int, default="0",
                        help='number of loading workers, needs to be 0 for Windows')
    return parser.parse_args()


class MCMCSampler:
    def __init__(self, model, img_shape, sample_size, num_classes, cbuffer_size=256):
        """
        MCMC sampler that uses SGLD.

        :param model: Neural network to use for modeling the energy function E_\theta
        :param img_shape: Image shape (height x width)
        :param sample_size: Number of images to sample
        :param num_classes: Number of output nodes, i.e., number of classes
        :param cbuffer_size: Size of the buffer per class the is being retained for reservoir sampling
        """
        super().__init__()
        self.model = model
        self.img_shape = img_shape
        self.sample_size = sample_size
        self.num_classes = num_classes
        self.cbuffer_size = cbuffer_size
        self.examples = [(torch.rand((1,) + img_shape) * 2 - 1) for _ in range(self.sample_size)] ##cbuffer*no output
        self.buffer_u = [(torch.rand((1,) + img_shape) * 2 - 1) for _ in range(self.sample_size)]
        self.buffer_c = {i: [(torch.rand((1,) + img_shape) * 2 - 1) for _ in range(self.sample_size)] for i in
                         range(self.num_classes)}
        self.from_buffer_list = np.random.choice([True, False], size=self.sample_size, p=[0.9, 0.1]).tolist()

    def synthesize_samples(self, clabel=None, steps=60, step_size=10, return_img_per_step=False):
        """
        Synthesize images from the current parameterized q_\theta

        :param model: Neural network to use to model E_theta
        :param clabel: Class label(s) used to sample the buffer
        :param steps: Number of iterations in the MCMC algorithm.
        :param step_size: Learning rate/update step size
        :param return_img_per_step: images during MCMC-based synthesis
        :return: synthesized images
        """
        # Before MCMC: set model parameters to "required_grad=False"
        # because we are only interested in the gradients of the input.
        is_training = self.model.training
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # Enable gradient calculation if not already the case
        had_gradients_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        # TODO (3.3): Implement SGLD-based synthesis with reservoir sampling

        # Sample initial data points x^0 to get a starting point for the sampling process.
        # As seen in the lecture and the theoretical recap, there exist multiple variants how we can approach this task.

        # --> Here, you should use non-persistent short-run MCMC and combine it with reservoir sampling. This means that
        # you sample a small portion of new images from random Gaussian noise, while the rest is taking from a buffer
        # that is re-populated at the end of synthesis.

        # In practical terms, you want to create a buffer that persists across epochs
        # (consider saving that into a field of this class). In this buffer, you store the synthesized samples after
        # each SGLD procedure. In the class-conditional setting, you want to have individual buffers per class.
        # Please make sure that you keep the buffer finite to not run into memory-related problems.
        ##define buffer size =len(self.ecxpmle) else cbufeer
        if clabel is None:
            # Unconditional case
            n_new = np.random.binomial(self.sample_size, 0.1)
            rand_imgs = torch.rand((n_new,) + self.img_shape) * 2 - 1
            old_imgs = torch.cat(random.choices(self.buffer_u, k=self.sample_size - n_new), dim=0)
            inp_imgs = torch.cat([rand_imgs, old_imgs], dim=0).to(device)
            # inp_imgs = torch.cat([rand_imgs, old_imgs], dim=0).to(device)
        else:
            # Conditional case
            inp_imgs_list = []
            for i, label in enumerate(clabel):
                if self.from_buffer_list[i]:
                    inp_imgs_class_prev = random.choices(self.buffer_c[label.item()])
                    inp_imgs_class = inp_imgs_class_prev[0]
                    # inp_imgs_class = torch.stack(inp_imgs_class)  # Convert list to tensor
                else:
                    inp_imgs_class = torch.rand((1,) + self.img_shape) * 2 - 1

                inp_imgs_list.append(inp_imgs_class)
            inp_imgs = torch.cat(inp_imgs_list, dim=0).to(device)

        #inp_imgs = None  # corresponds to the initial sample(s) x^0
        inp_imgs.requires_grad = True
        # print('GRADDDDD: ', inp_imgs.grad)

        # List for storing generations at each step
        imgs_per_step = []

        # Execute K MCMC steps
        for _ in range(steps):
            # (1) Add small noise to the input 'inp_imgs' (normlized to a range of -1 to 1).
            # This corresponds to the Brownian noise that allows to explore the entire parameter space.
            noise = torch.randn_like(inp_imgs)*0.005 ## same size of image noise  normal
            inp_imgs.data.add(noise.data)
            inp_imgs.data.clamp_(min=-1.0, max=1.0)

            # (2) Calculate gradient-based score function at the current step. In case of the JEM implementation AND
            # class-conditional sampling (which is optional from a methodological point of view), make sure that you
            # plug in some label information as well as we want to calculate E(x,y) and not only E(x).
            if clabel is not None:
                logits = -self.model(inp_imgs, clabel)
            else:
                logits = -self.model(inp_imgs)

            # (3) Perform gradient ascent to regions of higher probability
            #prob = torch.nn.Softmax(logits)
            # (gradient descent if we consider the energy surface!). You can use the parameter 'step_size' which can be
            # prob.sum().backward()
            # inp_imgs.grad.data.clamp_(-0.03, 0.03)
            # print(f'Logits: {logits}')
            # print('GRADDDDD: ', inp_imgs.grad)
            logits.sum().backward()
            # grad = torch.autograd.grad(torch.log(logits), inp_imgs)[0]
            inp_imgs.grad.data.clamp_(-0.03, 0.03)
            # considered the learning rate of the SGLD update.
            inp_imgs.data.add(-step_size * inp_imgs.grad.data)
            inp_imgs.grad.detach()
            inp_imgs.grad.zero_()
            inp_imgs.data.clamp_(min=-1.0, max=1.0)

            # (4) Optional: save (detached) intermediate images in the imgs_per_step variable
            if return_img_per_step:
                imgs_per_step.append(inp_imgs.clone())

        for p in self.model.parameters():
            p.requires_grad = True
        self.model.train(is_training)

        torch.set_grad_enabled(had_gradients_enabled)

        '''self.examples = list(inp_imgs.to(torch.device("cpu")).chunk(self.sample_size, dim=0)) + self.examples
        inp_imgs = self.examples[buffer_idx, torch.randint(0, self.cbuffer_size, (self.sample_size,))]'''
        #self.examples = self.examples[:self.cbuffer_size]

        if return_img_per_step:
            return torch.stack(imgs_per_step, dim=0)
        else:
            return inp_imgs


class JEM(pl.LightningModule):
    def __init__(self, img_shape, batch_size, num_classes=42, cbuffer_size=256, ccond_sample=False, num_epochs=5, alpha=0.1, lmbd=0.1,
                 lr=1e-4, lr_stepsize=1, lr_gamma=0.97, m_in=0, m_out=-10, steps=60, step_size_decay=1.0, **MODEL_args):
        super().__init__()
        self.save_hyperparameters()

        self.img_shape = img_shape
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.ccond_sample = ccond_sample
        # print(MODEL_args)
        self.cnn = ShallowCNN(**MODEL_args)

        # During training, we want to use the MCMC-based sampler to synthesize images from the current q_\theta and
        # use these in the contrastive loss functional to update the model parameters \theta.
        # (Intuitively, we alternate between sampling from q_\theta and updating q_\theta, which is a quite challenging
        # minmax setting with an adversarial interpretation.)
        self.sampler = MCMCSampler(self.cnn, img_shape=img_shape, sample_size=batch_size, num_classes=num_classes,
                                   cbuffer_size=cbuffer_size)
        self.example_input_array = torch.zeros(1, *img_shape)  # this is used to validate data and model compatability

        # If you want, you can use Torchmetrics to evaluate your classification performance!
        # For example, if we want to populate the metrics after each training step using the predicted logits and
        # classification ground truth y:
        #         self.train_metrics.update(logits, y) --> populate the running metrics buffer
        # We can then log the metrics using on_step=False and on_epoch=True so that they only get computed at the
        # end of each epoch.
        #         self.log_dict(self.train_metrics, on_step=False, on_epoch=True)
        # Please refer to the torchmetrics documentation if this process is not clear.
        metrics = torchmetrics.MetricCollection([torchmetrics.CohenKappa(num_classes=num_classes,task='multiclass'),
                                                 torchmetrics.AveragePrecision(num_classes=num_classes,task='multiclass'),
                                                 torchmetrics.AUROC(num_classes=num_classes,task='multiclass'),
                                                 torchmetrics.MatthewsCorrCoef(num_classes=num_classes,task='multiclass'),
                                                 torchmetrics.CalibrationError(task='multiclass',num_classes=num_classes)])
        dyna_metrics = [torchmetrics.Accuracy,
                        torchmetrics.Precision,
                        torchmetrics.Recall,
                        torchmetrics.Specificity,
                        torchmetrics.F1Score]

        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='val_')
        for mode in ['micro', 'macro']:
            self.train_metrics.add_metrics(
                {f"{mode}_{m.__name__}": m(average=mode, num_classes=num_classes,task='multiclass') for m in dyna_metrics})
            self.valid_metrics.add_metrics(
                {f"{mode}_{m.__name__}": m(average=mode, num_classes=num_classes,task='multiclass') for m in dyna_metrics})

        self.hp_metric = torchmetrics.AveragePrecision(num_classes=num_classes,task='multiclass')

    def forward(self, x, labels=None):
        z = self.cnn(x, labels)
        return z

    def configure_optimizers(self):
        # We typically do not want to have momentum enabled. This is because when training the EBM using alternating
        # steps of synthesis and model update, we constantly shift the energy surface, making it hard to make momentum
        # helpful.
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, betas=(0.0, 0.999))

        # Exponential decay over epochs
        scheduler = optim.lr_scheduler.StepLR(optimizer, self.hparams.lr_stepsize,
                                              gamma=self.hparams.lr_gamma)
        return [optimizer], [scheduler]

    def px_step(self, batch, ccond_sample=True):
        # TODO (3.4): Implement p(x) step
        real_images, labels = batch
        added_noise = torch.randn_like(real_images) *0.005
        real_images += added_noise
        real_images = real_images.clamp_(min =-1,max =1)
         # difference?

        if ccond_sample: ###conditional JEM on
            # sample randomly 0,numclsses,(batch_size)
            fake_images = self.sampler.synthesize_samples(clabel=batch[1])
            # scores = score_fn(model=self.cnn, x=total_image,y= labels, score="px")
        else:
            fake_images = self.sampler.synthesize_samples()

        cdiv_loss = real_images.mean() - fake_images.mean() ##swap
        reg_loss = self.hparams.alpha * (real_images ** 2 + fake_images ** 2).mean()
        loss = cdiv_loss + reg_loss
        return loss


    # reg_loss = self.hparams.alpha * scores.mean()
    # loss = reg_loss + cdiv_loss
    def pyx_step(self, batch):
        # TODO (3.4): Implement p(y|x) step
        # Here, we want to calculate the classification loss using the class logits infered by the neural network.
        images, labels = batch
        ##add noise immages
        added_noise = torch.randn_like(images) * 0.005
        images = images + added_noise
        #clamp
        images.data.clamp_(min=-1.0, max=1.0)
        ##logits cnn.getlogits
        # logits = score_fn(model=self.cnn, x=images, y=labels, score="py")
        logits = self.cnn.get_logits(images)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        ##update train matrix self.train_metrics.update(logits,real_y)
        self.train_metrics.update(logits,labels)
        return loss

    def training_step(self, batch, batch_idx):
        # TODO (3.4): Implement joint density p(x,y) step using p(x) and p(y|x)
        px_loss = self.px_step(batch,ccond_sample= self.ccond_sample)
        pyx_loss = self.pyx_step(batch)
        loss = px_loss + pyx_loss
        self.log("train_loss", loss) ##if we learn against loss then it will learn joint probability
        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=None):
        real_imgs, real_label = batch
        fake_imgs = torch.rand_like(real_imgs) * 2 - 1

        inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0)
        real_out, fake_out = self.cnn(inp_imgs).chunk(2, dim=0)

        cdiv = fake_out.mean() - real_out.mean()
        #logits for real_images
        logits = self.cnn.get_logits(real_imgs)
        ##update hp metric
        self.hp_metric(logits, real_label)
        #logg loss
        self.log("val_contrastive_divergence", cdiv)
        return cdiv


def run_training(args) -> pl.LightningModule:
    """
    Perform EBM/JEM training using a set of hyper-parameters

    Visualization can be either done showcasing different image states during synthesis or by showcasing the
    final results.

    :param args: hyper-parameter
    :return: pl.LightningModule: the trained model
    """
    # Hyper-parameters
    ckpt_dir = args.ckpt_dir
    data_dir = args.data_dir
    num_workers = args.num_workers  # 0 for Windows, can be set higher for linux
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    num_classes = args.num_classes
    lr = args.lr
    lr_stepsize = args.lr_stepsize
    lr_gamma = args.lr_gamma
    alpha = args.alpha
    cbuffer_size = args.cbuffer_size
    ccond_sample = args.ccond_sample

    # Create checkpoint path if it doesn't exist yet
    os.makedirs(ckpt_dir, exist_ok=True)

    # Datasets & Dataloaders
    # print(data_dir)
    datasets: Dict[str, TransformTensorDataset] = get_datasets(data_dir)
    train_loader = data.DataLoader(datasets['train'], batch_size=batch_size, shuffle=True, drop_last=True,
                                   num_workers=num_workers, pin_memory=True)
    val_loader = data.DataLoader(datasets['val'], batch_size=batch_size, shuffle=False, drop_last=False,
                                 num_workers=num_workers)

    trainer = pl.Trainer(default_root_dir=ckpt_dir,
                         #gpus=1 if str(device).startswith("cuda") else 0,
                         max_epochs=num_epochs,
                         gradient_clip_val=0.1,
                         callbacks=[
                             ModelCheckpoint(save_weights_only=True, mode="min", monitor='val_contrastive_divergence',
                                             filename='val_condiv_{epoch}-{step}', every_n_epochs=2),
                             # ModelCheckpoint(save_weights_only=True, mode="max", monitor='val_MulticlassAveragePrecision',
                             #                 filename='val_mAP_{epoch}-{step}'),
                             ModelCheckpoint(save_weights_only=True, filename='last_{epoch}-{step}'),
                             LearningRateMonitor("epoch")
                         ])
    pl.seed_everything(42)
    model = JEM(num_epochs=num_epochs,
                img_shape=(1, 56, 56),  # shape of the images (channels, height, width)
                batch_size=batch_size,
                num_classes=num_classes,
                hidden_features=32,  # size of the hidden dimension in the Shallow CNN model
                cbuffer_size=cbuffer_size,  # size of the reservoir for sampling (class-specific)
                ccond_sample=ccond_sample,  # Should we do class-conditional sampling?
                lr=lr,  # General Learning rate
                lr_gamma=lr_gamma,  # Multiplicative factor for exponential learning rate decay
                lr_stepsize=lr_stepsize,  # Step size for exponential learning rate decay
                alpha=alpha,  # L2 regularization of energy terms
                step_size_decay=1.0  # Multiplicative factor for SGLD step size decay)
                )
    trainer.fit(model, train_loader, val_loader)
    model = JEM.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    return model


def run_generation(args, ckpt_path: Union[str, Path], conditional: bool = False):
    """
    With a trained model we can synthesize new examples from q_\theta using SGLD.

    :param args: hyper-parameter
    :param ckpt_path: local path to the trained checkpoint.
    :param conditional: flag to specify if we want to generate conditioned on a specific class label or not
    :return: None
    """
    model = JEM.load_from_checkpoint(ckpt_path)
    model.to(device)
    pl.seed_everything(42)

    def gen_imgs(model, clabel=None, step_size=10, batch_size=24, num_steps=256):
        model.eval()
        torch.set_grad_enabled(True)  # Tracking gradients for sampling necessary
        mcmc_sampler = MCMCSampler(model, model.img_shape, batch_size, model.num_classes)
        img = mcmc_sampler.synthesize_samples(clabel, steps=num_steps, step_size=step_size, return_img_per_step=True)
        torch.set_grad_enabled(False)
        model.train()
        return img

    k = 8
    bs = 8
    num_steps = 256
    conditional_labels = [1, 4, 5, 10, 17, 18, 39, 23]

    synth_imgs = []
    for label in tqdm.tqdm(conditional_labels):
        clabel = (torch.ones(bs) * label).type(torch.LongTensor).to(model.device)
        generated_imgs = gen_imgs(model, clabel=clabel if conditional else None, step_size=10, batch_size=bs, num_steps=num_steps).cpu()
        synth_imgs.append(generated_imgs[-1])

        # Visualize sampling process
        i = 0
        step_size = num_steps // 8
        imgs_to_plot = generated_imgs[step_size - 1::step_size, i]
        imgs_to_plot = torch.cat([generated_imgs[0:1, i], imgs_to_plot], dim=0)
        grid = torchvision.utils.make_grid(imgs_to_plot, nrow=imgs_to_plot.shape[0], normalize=True,
                                           value_range=(-1, 1), pad_value=0.5, padding=2)
        grid = grid.permute(1, 2, 0)
        plt.figure(figsize=(8, 8))
        plt.imshow(grid)
        plt.xlabel("Generation iteration")
        plt.xticks([(generated_imgs.shape[-1] + 2) * (0.5 + j) for j in range(8 + 1)],
                   labels=[1] + list(range(step_size, generated_imgs.shape[0] + 1, step_size)))
        plt.yticks([])
        plt.savefig(f"{'conditional' if conditional else 'unconditional'}_sample_label={label}.png")

    # Visualize end results
    grid = torchvision.utils.make_grid(torch.cat(synth_imgs), nrow=k, normalize=True, value_range=(-1, 1),
                                       pad_value=0.5,
                                       padding=2)
    grid = grid.permute(1, 2, 0)
    grid = grid[..., 0].numpy()
    plt.figure(figsize=(12, 24))
    plt.imshow(grid, cmap='Greys')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f"{'conditional' if conditional else 'unconditional'}_samples.png")


def run_evaluation(args, ckpt_path: Union[str, Path]):
    """
    Evaluate the predictive performance of the JEM model.
    :param args: hyper-parameter
    :param ckpt_path: local path to the trained checkpoint.
    :return: None
    """
    model = JEM.load_from_checkpoint(ckpt_path)
    model.to(device)
    pl.seed_everything(42)

    # Datasets & Dataloaders
    batch_size = args.batch_size
    data_dir = args.data_dir
    num_workers = args.num_workers
    datasets: Dict[str, TransformTensorDataset] = get_datasets(data_dir)

    # Test loader
    test_loader = data.DataLoader(datasets['test'], batch_size=batch_size, shuffle=False, drop_last=False,
                                  num_workers=num_workers)

    trainer = pl.Trainer() #gpus=1 if str(device).startswith("cuda") else 0)
    results = trainer.validate(model, dataloaders=test_loader)
    print(results)
    return results


def run_ood_analysis(args, ckpt_path: Union[str, Path]):
    """
    Run out-of-distribution (OOD) analysis. First, you evaluate the scores for the training samples (in-distribution),
    a random noise distribution, and two different distributions that share some resemblence with the training data.

    :param args: hyper-parameter
    :param ckpt_path: local path to the trained checkpoint.
    :return: None
    """
    model = JEM.load_from_checkpoint(ckpt_path)
    model.to(device)
    pl.seed_everything(42)

    # Datasets & Dataloaders
    batch_size = args.batch_size
    data_dir = args.data_dir
    num_workers = args.num_workers
    datasets: Dict[str, TransformTensorDataset] = get_datasets(data_dir)

    # Test loader
    test_loader = data.DataLoader(datasets['test'], batch_size=batch_size, shuffle=False, drop_last=False,
                                  num_workers=num_workers)
    # OOD loaders for OOD types a and b
    ood_ta_loader = data.DataLoader(datasets['ood_ta'], batch_size=batch_size, shuffle=False, drop_last=False,
                                    num_workers=num_workers)
    ood_tb_loader = data.DataLoader(datasets['ood_tb'], batch_size=batch_size, shuffle=False, drop_last=False,
                                    num_workers=num_workers)

    # TODO (3.6): Calculate and visualize the score distributions, e.g. with a histogram. Analyze whether we can
    #  visualy tell apart the different data distributions based on their assigned score.

    # TODO (3.6): Solve a binary classification on the soft scores and evaluate and AUROC and/or AUPRC score for
    #  discrimination between the training samples and one of the OOD distributions.
    for x, y in test_loader:
        test_score = []
        x = x.clamp(min=-1, max=1).to(model.device)
        y = y.to(model.device)
        out = score_fn(model=model, x=x, y=y, score="px")
        test_score.append(out)

    for x, y in ood_ta_loader:
        ood_a_score = []
        x = x.clamp(min=-1, max=1).to(model.device)
        y = y.to(model.device)
        out = score_fn(model=model, x=x, y=y, score="px")
        ood_a_score.append(out)

    for x, y in ood_tb_loader:
        ood_b_score = []
        x = x.clamp(min=-1, max=1).to(model.device)
        y = y.to(model.device)
        out = score_fn(model=model, x=x, y=y, score="px")
        ood_b_score.append(out)

    plt.hist(test_score, bins=50, alpha=0.5, label='test_score')
    plt.hist(ood_a_score, bins=50, alpha=0.5, label='ood_a_score')
    plt.hist(ood_b_score, bins=50, alpha=0.5, label='ood_b_score')
    plt.xlabel('Scores')
    plt.ylabel('Frequency')
    plt.title('Distribution of Scores')
    plt.legend()

    save_path = '/home/cip/ai2022/qi27ycyt/plots/histogram.png'
    plt.savefig(save_path)
    plt.show()

    ## AUC
    labels_test = torch.zeros(len(test_score)) # Assign label 0 to test samples
    total_OOD_score = torch.cat([ood_a_score,ood_b_score],dim=0)
    labels_ood = torch.ones(total_OOD_score.shape[0])  # Assign label 1 to OOD samples

    labels = torch.cat([labels_test, labels_ood], dim=0)

    # Calculate AUROC score
    auroc = roc_auc_score(labels.cpu(), total_OOD_score.cpu())


if __name__ == '__main__':
    args = parse_args()

    # 1) Run training
    #run_training(args)

    # 2) Evaluate model
    last_ckpt_dir = args.ckpt_dir + "lightning_logs/version_4/checkpoints/"
    file_path = next((os.path.join(last_ckpt_dir, file) for file in os.listdir(last_ckpt_dir) if file.startswith("last")), None)
    ckpt_path: str = file_path

    # Classification performance
    run_evaluation(args, ckpt_path)

    # Image synthesis
    run_generation(args, ckpt_path, conditional=True)
    run_generation(args, ckpt_path, conditional=False)

    # OOD Analysis
    run_ood_analysis(args, ckpt_path)
