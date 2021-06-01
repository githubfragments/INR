'''Implements a generic training loop.
'''

import torch
from siren import utils
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import time
import numpy as np
import os
import shutil
from losses import model_l1, spectral_norm_loss

def train(model, train_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn,
          summary_fn, val_dataloader=None, double_precision=False, clip_grad=False, use_lbfgs=False, loss_schedules=None,
          weight_decay=0, l1_reg=0,  l1_loss_fn=model_l1,  spec_reg=0):

    optim = torch.optim.Adam(lr=lr, params=model.parameters(), weight_decay=weight_decay)

    # copy settings from Raissi et al. (2019) and here 
    # https://github.com/maziarraissi/PINNs
    if use_lbfgs:
        optim = torch.optim.LBFGS(lr=lr, params=model.parameters(), max_iter=50000, max_eval=50000,
                                  history_size=50, line_search_fn='strong_wolfe')

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
    use_amp = True

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

 # set_to_none=True here can modestly improve performance
    total_steps = 0
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        best_mse = 1
        for epoch in range(epochs):
            if not epoch % epochs_til_checkpoint and epoch:
                torch.save(model.state_dict(),
                           os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % epoch),
                           np.array(train_losses))

            for step, (model_input, gt) in enumerate(train_dataloader):
                with torch.cuda.amp.autocast(enabled=use_amp):
                    start_time = time.time()

                    model_input = {key: value.cuda() for key, value in model_input.items()}
                    gt = {key: value.cuda() for key, value in gt.items()}

                    if double_precision:
                        model_input = {key: value.double() for key, value in model_input.items()}
                        gt = {key: value.double() for key, value in gt.items()}

                    if use_lbfgs:
                        def closure():
                            optim.zero_grad()
                            model_output = model(model_input)
                            losses = loss_fn(model_output, gt)
                            train_loss = 0.
                            for loss_name, loss in losses.items():
                                train_loss += loss.mean()
                            train_loss.backward()
                            return train_loss

                        optim.step(closure)

                    model_output = model(model_input)
                    losses = loss_fn(model_output, gt)
                    if l1_reg > 0:
                        l1_loss = l1_loss_fn(model, l1_reg)
                        losses = {**losses, **l1_loss}
                    if spec_reg > 0:
                        spec_loss = spectral_norm_loss(model, spec_reg)
                        losses = {**losses, **spec_loss}


                    train_loss = 0.
                    for loss_name, loss in losses.items():
                        single_loss = loss.mean()

                        if loss_schedules is not None and loss_name in loss_schedules:
                            writer.add_scalar(loss_name + "_weight", loss_schedules[loss_name](total_steps), total_steps)
                            single_loss *= loss_schedules[loss_name](total_steps)

                        writer.add_scalar(loss_name, single_loss, total_steps)
                        train_loss += single_loss

                    train_losses.append(train_loss.item())
                    writer.add_scalar("total_train_loss", train_loss, total_steps)

                    if train_loss < best_mse:
                        best_mse = train_loss
                        best_state_dict = model.state_dict()

                    if not total_steps % steps_til_summary:
                        torch.save(model.state_dict(),
                                   os.path.join(checkpoints_dir, 'model_current_.pth'))
                        summary_fn(model, model_input, gt, model_output, writer, total_steps)

                    if not use_lbfgs:
                        optim.zero_grad()
                        scaler.scale(train_loss).backward()


                        if clip_grad:
                            if isinstance(clip_grad, bool):
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                            else:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

                        scaler.step(optim)
                        scaler.update()

                    pbar.update(1)

                    if not total_steps % steps_til_summary:
                        tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))

                        if val_dataloader is not None:
                            print("Running validation set...")
                            model.eval()
                            with torch.no_grad():
                                val_losses = []
                                for (model_input, gt) in val_dataloader:
                                    model_output = model(model_input)
                                    val_loss = loss_fn(model_output, gt)
                                    val_losses.append(val_loss)

                                writer.add_scalar("val_loss", np.mean(val_losses), total_steps)
                            model.train()

                total_steps += 1

        torch.save(model.state_dict(),
                   os.path.join(checkpoints_dir, 'model_final.pth'))
        torch.save(best_state_dict,
                   os.path.join(checkpoints_dir, 'model_best_.pth'))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
                   np.array(train_losses))

        model.load_state_dict(best_state_dict, strict=True)

def train_phased(model, train_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn,
          summary_fn, val_dataloader=None, double_precision=False, clip_grad=False, use_lbfgs=False, loss_schedules=None,
          weight_decay=0, l1_reg=0, spec_reg=0, phased=False, intermediate_losses=False):

    optim = torch.optim.Adam(lr=lr, params=model.parameters(), weight_decay=weight_decay)

    # copy settings from Raissi et al. (2019) and here
    # https://github.com/maziarraissi/PINNs
    if use_lbfgs:
        optim = torch.optim.LBFGS(lr=lr, params=model.parameters(), max_iter=50000, max_eval=50000,
                                  history_size=50, line_search_fn='strong_wolfe')

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
    use_amp = True

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_mse = float('inf')

    for num_phase, hidden_dim in enumerate(model.hidden_features):
        if (not phased) and num_phase + 1 < len(model.hidden_features): continue


        # Deactivate grad for all parameters
        if phased:
            for param in model.parameters():
                param.requires_grad = False
            #Activate grad only for the net to be trained
            trained_net = model.nets[num_phase]
            for param in trained_net.parameters():
                param.requires_grad = True
        total_steps = 0
        print(f'Phase {num_phase}')
        with tqdm(total=len(train_dataloader) * epochs) as pbar:
            train_losses = []

            for epoch in range(epochs):
                if not epoch % epochs_til_checkpoint and epoch:
                    torch.save(model.state_dict(),
                               os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
                    np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % epoch),
                               np.array(train_losses))

                for step, (model_input, gt) in enumerate(train_dataloader):
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        start_time = time.time()

                        model_input = {key: value.cuda() for key, value in model_input.items()}
                        gt = {key: value.cuda() for key, value in gt.items()}

                        if double_precision:
                            model_input = {key: value.double() for key, value in model_input.items()}
                            gt = {key: value.double() for key, value in gt.items()}

                        if use_lbfgs:
                            def closure():
                                optim.zero_grad()
                                model_output = model(model_input)[:num_phase + 1]
                                model_output = {'model_in': model_output[0]['model_in'],
                                                'model_out': sum([p['model_out'] for p in model_output])}
                                losses = loss_fn(model_output, gt)
                                train_loss = 0.
                                for loss_name, loss in losses.items():
                                    train_loss += loss.mean()
                                train_loss.backward()
                                return train_loss

                            optim.step(closure)
                        losses = {}
                        if (not phased) and intermediate_losses:

                            for i in range(num_phase):
                                model_output = model(model_input)[:i + 1]
                                model_output = {'model_in': model_output[0]['model_in'],
                                                'model_out': sum([p['model_out'] for p in model_output])}
                                intermediate_loss = loss_fn(model_output, gt)
                                intermediate_loss['img_loss_intermediate' + str(i)] = intermediate_loss.pop('img_loss')
                                losses = {**losses, **intermediate_loss}

                        model_output = model(model_input)[:num_phase+1]
                        model_output = {'model_in': model_output[0]['model_in'], 'model_out': sum([p['model_out'] for p in model_output])}
                        combined_loss = loss_fn(model_output, gt)
                        losses = {**losses, **combined_loss}
                        if l1_reg > 0:
                            l1_loss = model_l1(model, l1_reg)
                            losses = {**losses, **l1_loss}
                        if spec_reg > 0:
                            spec_loss = spectral_norm_loss(model, spec_reg)
                            losses = {**losses, **spec_loss}


                        train_loss = 0.
                        for loss_name, loss in losses.items():
                            single_loss = loss.mean()

                            if loss_schedules is not None and loss_name in loss_schedules:
                                writer.add_scalar(loss_name + "_weight", loss_schedules[loss_name](total_steps), total_steps)
                                single_loss *= loss_schedules[loss_name](total_steps)

                            writer.add_scalar(loss_name, single_loss, total_steps)
                            train_loss += single_loss

                        train_losses.append(train_loss.item())
                        writer.add_scalar("total_train_loss", train_loss, total_steps)
                        if losses['img_loss'] < best_mse and num_phase + 1 == len(model.hidden_features):
                            best_mse = losses['img_loss']
                            best_state_dict = model.state_dict()

                        if not total_steps % steps_til_summary:
                            torch.save(model.state_dict(),
                                       os.path.join(checkpoints_dir, 'model_current_.pth'))
                            summary_fn(model, model_input, gt, model_output, writer, total_steps)

                        if not use_lbfgs:
                            optim.zero_grad()
                            scaler.scale(train_loss).backward()


                            if clip_grad:
                                if isinstance(clip_grad, bool):
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                                else:
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

                            scaler.step(optim)
                            scaler.update()

                        pbar.update(1)

                        if not total_steps % steps_til_summary:
                            tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))

                            if val_dataloader is not None:
                                print("Running validation set...")
                                model.eval()
                                with torch.no_grad():
                                    val_losses = []
                                    for (model_input, gt) in val_dataloader:
                                        model_output = model(model_input)
                                        val_loss = loss_fn(sum(model_output[:num_phase]), gt)
                                        val_losses.append(val_loss)

                                    writer.add_scalar("val_loss", np.mean(val_losses), total_steps)
                                model.train()

                    total_steps += 1

    torch.save(model.state_dict(),
               os.path.join(checkpoints_dir, 'model_final.pth'))
    torch.save(best_state_dict,
               os.path.join(checkpoints_dir, 'model_best_.pth'))
    np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
               np.array(train_losses))
class LinearDecaySchedule():
    def __init__(self, start_val, final_val, num_steps):
        self.start_val = start_val
        self.final_val = final_val
        self.num_steps = num_steps

    def __call__(self, iter):
        return self.start_val + (self.final_val - self.start_val) * min(iter / self.num_steps, 1.)
