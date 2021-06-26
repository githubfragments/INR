'''Implements a generic training loop.
'''
import matplotlib.pyplot as plt
import torch
from siren import utils
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import time
import numpy as np
import os
import shutil
from losses import model_l1, spectral_norm_loss
import learn2learn as l2l

def train(model, train_dataloader, maml_iterations, inner_lr, outer_lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn,
          summary_fn, val_dataloader=None, double_precision=False, clip_grad=False, use_lbfgs=False, loss_schedules=None,
          weight_decay=0, l1_reg=0, l1_loss_fn=model_l1, spec_reg=0,  maml_batch_size=None, maml_adaptation_steps=None):

    # def compute_loss(model):
    #     model_output = model(model_input)
    #     losses = loss_fn(model_output, gt)
    #     if l1_reg > 0:
    #         l1_loss = l1_loss_fn(model, l1_reg)
    #         losses = {**losses, **l1_loss}
    #     if spec_reg > 0:
    #         spec_loss = spectral_norm_loss(model, spec_reg)
    #         losses = {**losses, **spec_loss}
    #
    #     train_loss = 0.
    #     for loss_name, loss in losses.items():
    #         single_loss = loss.mean()
    #
    #         if loss_schedules is not None and loss_name in loss_schedules:
    #             writer.add_scalar(loss_name + "_weight", loss_schedules[loss_name](total_steps), total_steps)
    #             single_loss *= loss_schedules[loss_name](total_steps)
    #
    #         writer.add_scalar(loss_name, single_loss, total_steps)
    #         train_loss += single_loss
    #     return train_loss, model_output

    maml = l2l.algorithms.MAML(model, lr=inner_lr)
    opt = torch.optim.Adam(lr=outer_lr, params=maml.parameters(), weight_decay=weight_decay)


    if os.path.exists(model_dir):
        #val = input("The model directory %s exists. Overwrite? (y/n)"%model_dir)
        val = 'y'
        if val == 'y':
            shutil.rmtree(model_dir)

    else:
        os.makedirs(model_dir)

    summaries_dir = os.path.join(model_dir, 'summaries')
    utils.cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    utils.cond_mkdir(checkpoints_dir)

    writer = SummaryWriter(summaries_dir)

    num_iterations = maml_iterations
    meta_batch_size = maml_batch_size

    def fast_adapt(batch, learner, adaptation_steps):
        model_input, gt = batch
        model_input = {key: value.cuda() for key, value in model_input.items()}
        gt = {key: value.cuda() for key, value in gt.items()}


        # Adapt the model
        for step in range(adaptation_steps):
            losses = loss_fn(learner(model_input), gt)
            train_error = 0
            for loss_name, loss in losses.items():
                single_loss = loss.mean()
                train_error += single_loss
            learner.adapt(train_error)

        # Evaluate the adapted model
        predictions = learner(model_input)

        losses = loss_fn(predictions, gt)
        valid_error = 0
        for loss_name, loss in losses.items():
            single_loss = loss.mean()
            valid_error += single_loss
        return valid_error

 # set_to_none=True here can modestly improve performance
 #    total_steps = 0
 #    fig = plt.figure(1)
 #    im = plt.imshow(np.random.rand(256, 384, 3))
 #    plt.show()
    with tqdm(total=len(train_dataloader) * maml_iterations) as pbar:
        train_losses = []
        best_mse = 1
        for iteration in range(num_iterations):
            opt.zero_grad()
            meta_train_error = 0.0
            meta_train_accuracy = 0.0
            meta_valid_error = 0.0
            meta_valid_accuracy = 0.0
            for step, batch in enumerate(train_dataloader):
                # Compute meta-training loss
                learner = maml.clone()
                evaluation_error = fast_adapt(batch, learner, maml_adaptation_steps)
                evaluation_error.backward()
                meta_train_error += evaluation_error.item()


            # Average the accumulated gradients and optimize
            for p in maml.parameters():
                p.grad.data.mul_(1.0 / meta_batch_size)
            opt.step()
            pbar.update(1)



            if not iteration % epochs_til_checkpoint:
                # model_input, gt = batch
                # model_input = {key: value.cuda() for key, value in model_input.items()}
                # imout = model(model_input)
                # plt.figure(1)
                # plt.imshow(imout['model_out'].cpu().detach().reshape((256, 384, 3)).numpy()/2 + 0.5)
                # plt.show()
                torch.save(model.state_dict(),
                           os.path.join(checkpoints_dir, 'model_maml_iteration{}.pth'.format(iteration)))
                tqdm.write("Iteration %d, Meta Train Error %0.6f" % (iteration, meta_train_error))



            #     np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % epoch),
            #                np.array(train_losses))
            #
            # for step, (model_input, gt) in enumerate(train_dataloader):
            #     with torch.cuda.amp.autocast(enabled=False):
            #         start_time = time.time()
            #
            #         model_input = {key: value.cuda() for key, value in model_input.items()}
            #         gt = {key: value.cuda() for key, value in gt.items()}
            #
            #         if double_precision:
            #             model_input = {key: value.double() for key, value in model_input.items()}
            #             gt = {key: value.double() for key, value in gt.items()}
            #         optim.zero_grad()
            #         task_model = maml.clone()  # torch.clone() for nn.Modules
            #         adaptation_loss, _ = compute_loss(task_model)
            #         task_model.adapt(adaptation_loss)  # computes gradient, update task_model in-place
            #         evaluation_loss, model_output = compute_loss(task_model)
            #         evaluation_loss.backward()
            #         optim.step()
            #
            #
            #         writer.add_scalar("total_adaptation_loss", adaptation_loss, total_steps)
            #         writer.add_scalar("total_evaluation_loss", evaluation_loss, total_steps)
            #
            #
            #         if not total_steps % steps_til_summary:
            #              torch.save(model.state_dict(),
            #                         os.path.join(checkpoints_dir, 'model_maml_current_.pth'))
            #              summary_fn(model, model_input, gt, model_output, writer, total_steps)
            #
            #
            #         pbar.update(1)
            #
            #         if not total_steps % steps_til_summary:
            #             tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, evaluation_loss, time.time() - start_time))
            #         total_steps += 1
            # val_dataloader = train_dataloader
            # if val_dataloader is not None:
            #     model.eval()
            #     with torch.no_grad():
            #         val_losses = []
            #         for (model_input, gt) in val_dataloader:
            #             model_input = {key: value.cuda() for key, value in model_input.items()}
            #             gt = {key: value.cuda() for key, value in gt.items()}
            #             model_output = model(model_input)
            #             val_loss = loss_fn(model_output, gt)['img_loss'].cpu().detach().numpy()
            #             val_losses.append(val_loss)
            #         loss_mean = np.mean(val_losses)
            #         writer.add_scalar("val_loss", loss_mean, total_steps)
            #         print('Validation loss ', loss_mean )
            #     model.train()

        torch.save(model.state_dict(),
                   os.path.join(checkpoints_dir, 'model_maml.pth'))



