"""Train the model"""

import argparse
import logging
from tensorboardX import SummaryWriter
import os, shutil

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.optim as optim
import torchvision.models as models
from torch.autograd import Variable
from tqdm import tqdm
# from torchsummary import summary

import utils
import json
import model.net as net
import model.data_loader as data_loader
from evaluate import evaluate, evaluate_predictions

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='data', help="Directory containing the dataset")
parser.add_argument('--model-dir', default='experiments', help="Directory containing params.json")
parser.add_argument('--setting-dir', default='settings', help="Directory with different settings")
parser.add_argument('--setting', default='collider-pf', help="Directory contain setting.json, experimental setting, data-generation, regression model etc")
parser.add_argument('--fase', default='xybn', help='fase of training model, see manuscript for details. x, y, xy, bn, or feature')
parser.add_argument('--experiment', default='', help="Manual name for experiment for logging, will be subdir of setting")
parser.add_argument('--restore-file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'
parser.add_argument('--restore-last', action='store_true', help="continue a last run")
parser.add_argument('--restore-warm', action='store_true', help="continue on the run called 'warm-start.pth'")
parser.add_argument('--use-last', action="store_true", help="use last state dict instead of 'best' (use for early stopping manually)")
parser.add_argument('--cold-start', action='store_true', help="ignore previous state dicts (weights), even if they exist")
parser.add_argument('--warm-start', dest='cold_start', action='store_false', help="start from previous state dict")
parser.add_argument('--disable-cuda', action='store_true', help="Disable Cuda")
parser.add_argument('--no-parallel', action="store_false", help="no multiple GPU", dest="parallel")
parser.add_argument('--parallel', action="store_true", help="multiple GPU", dest="parallel")
parser.add_argument('--intercept', action="store_true", help="dummy run for getting intercept baseline results")
parser.add_argument('--visdom', action='store_true', help='generate plots with visdom')
parser.add_argument('--novisdom', dest='visdom', action='store_false', help='dont plot with visdom')
parser.add_argument('--monitor-grads', action='store_true', help='keep track of mean norm of gradients')
parser.set_defaults(parallel=False, cold_start=True, use_last=False, intercept=False, restore_last=False, save_preds=False,
                    monitor_grads=False, restore_warm=False, visdom=False)

def train(model, optimizer, loss_fn, dataloader, metrics, params, setting, writer=None, epoch=None, 
          mines=None, optims_mine=None):
    """Train the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """
    global train_tensor_keys, logdir

    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()

    # create storate for tensors for OLS after minibatches
    ts = []
    Xs = []
    Xtrues = []
    Ys = []
    Xhats = []
    Yhats = []
    Zhats = []

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as progress_bar:
        for i, batch in enumerate(dataloader):
            summary_batch = {}
            # put batch on cuda
            batch = {k: v.to(params.device) for k, v in batch.items()}
            if not (setting.covar_mode and epoch > params.suppress_t_epochs):
                batch["t"] = torch.zeros_like(batch['t'])
            Xs.append(batch['x'].detach().cpu())
            Xtrues.append(batch['x_true'].detach().cpu())

            # compute model output and loss
            output_batch = model(batch['image'], batch['t'].view(-1,1), epoch)
            Yhats.append(output_batch['y'].detach().cpu())

            # calculate loss
            if args.fase == "feature":
                # calculate loss for z directly, to get clear how well this can be measured
                loss_fn_z = torch.nn.MSELoss()
                loss_z = loss_fn_z(output_batch["y"].squeeze(), batch["z"])
                loss   = loss_z
                summary_batch["loss_z"] = loss_z.item()
            else:
                loss_fn_y = torch.nn.MSELoss()
                loss_y = loss_fn_y(output_batch["y"].squeeze(), batch["y"])
                loss   = loss_y
                summary_batch["loss_y"] = loss_y.item()

            # calculate other losses based on estimation of x
            if params.use_mi:
                mi_losses = {}
                # MINE mutual information calculate bottleneck loss
                for (mi_name, mi_estimator), mi_optim in zip(mines.items(), optims_mine.values()):
                    if 'monitor' in mi_name:                        
                        bottleneck_name = mi_name.split("_")[1]
                        target_name     = mi_name.split("_")[2]
                    else:
                        bottleneck_name = mi_name.split("_")[0]
                        target_name     = mi_name.split("_")[1]
                    mi_bn           = output_batch[bottleneck_name]
                    if "bn" in target_name:
                        mi_target = output_batch[target_name]
                    else:
                        mi_target = batch[target_name].view(-1,1)
                    for _ in range(params.num_mi_steps):
                        # update the MI estimator network for n steps
                        mi_loss = mi_estimator.lower_bound(mi_bn.detach(), mi_target.detach())
                        mi_optim.zero_grad()
                        mi_loss.backward(retain_graph=True)
                        mi_optim.step()

                    # after updating mi network, calculate MI for downward loss
                    mi_loss = mi_estimator.lower_bound(mi_bn, mi_target)
                    mi_losses[mi_name] = mi_loss

                    # store mutual information
                    summary_batch["mi_" + mi_name]  = -1*mi_loss.item()

                    # calculate spearman rho 
                    if mi_bn.shape[1] == 1:
                        summary_batch[mi_name + "_rho"] = net.spearmanrho(mi_target.detach().cpu(), mi_bn.detach().cpu())



            # calculate loss for colllider x
            if params.loss_x_type == 'mi':
                loss_x = mi_losses['bnx_x']
            elif params.loss_x_type == 'least-squares':
                # if not using mutual information to make bottleneck layer close to x, directly predict x with the CNN
                loss_fn_x = torch.nn.MSELoss()
                loss_x = loss_fn_x(output_batch["bnx"].squeeze(), batch["x"])
            else:
                raise NotImplementedError(f'x loss not implemented: {params.loss_x_type}, should be in mi, least-squares')
            summary_batch["loss_x"] = loss_x.item()
            if not params.alpha == 1:
                # possibly weigh down contribution of estimating x
                loss_x *= params.alpha
                summary_batch["loss_x_weighted"] = loss_x.item()
            # add x loss to total loss
            loss += loss_x            

            # add least squares regression on final layer
            if params.do_least_squares:
                X    = batch["x"].view(-1,1)
                t    = batch["t"].view(-1,1)
                Z    = output_batch["bnz"]
                if Z.ndimension() == 1:
                    Z.unsqueeze_(1)
                Xhat = output_batch["bnx"]
                # add intercept
                Zi = torch.cat([torch.ones_like(t), Z], 1)
                # add treatment info
                Zt = torch.cat([Zi, t], 1)
                Y  = batch["y"].view(-1,1)

                # regress y on final layer, without x
                betas_y = net.cholesky_least_squares(Zt, Y, intercept=False)
                y_hat   = Zt.matmul(betas_y).view(-1,1)
                mse_y  = ((Y - y_hat)**2).mean()

                summary_batch["regr_b_t"] = betas_y[-1].item()
                summary_batch["regr_loss_y"] = mse_y.item()

                # regress x on final layer without x
                betas_x = net.cholesky_least_squares(Zi, Xhat, intercept=False)
                x_hat   = Zi.matmul(betas_x).view(-1,1)
                mse_x  = ((Xhat - x_hat)**2).mean()

                # store all tensors for single pass after epoch
                Xhats.append(Xhat.detach().cpu())
                Zhats.append(Z.detach().cpu())
                ts.append(t.detach().cpu())
                Ys.append(Y.detach().cpu())

                summary_batch["regr_loss_x"] = mse_x.item()

            # add loss_bn only after n epochs
            if params.bottleneck_loss and epoch > params.bn_loss_lag_epochs:
                # only add to loss when bigger than margin
                if params.bn_loss_type == "regressor-least-squares":
                    if params.bn_loss_margin_type == "dynamic-mean":
                        # for each batch, calculate loss of just using mean for predicting x
                        mse_x_mean = ((X - X.mean())**2).mean()
                        loss_bn = torch.max(torch.zeros_like(mse_x), mse_x_mean - mse_x)
                    elif params.bn_loss_margin_type == "fixed":
                        mse_diff = params.bn_loss_margin - mse_x
                        loss_bn = torch.max(torch.zeros_like(mse_x), mse_diff)
                    else:
                        raise NotImplementedError(f'bottleneck loss margin type not implemented: {params.bn_loss_margin_type}')
                elif params.bn_loss_type == 'mi':
                    loss_bn = -1*mi_losses[params.bn_loss_mi]
                    #loss_bn = torch.max(torch.ones_like(loss_bn)*params.bn_loss_margin, loss_bn)
                else:
                    raise NotImplementedError(f'currently not implemented bottleneck loss type: {params.bn_loss_type}')

                # possibly reweigh bottleneck loss and add to total loss
                summary_batch["loss_bn"] = loss_bn.item()
                # note is this double?
                if loss_bn > params.bn_loss_margin:
                    loss_bn *= params.bottleneck_loss_wt
                    loss    += loss_bn

            # perform parameter update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            summary_batch['loss'] = loss.item()
            summ.append(summary_batch)

            # if necessary, write out tensors
            if params.monitor_train_tensors and (epoch % params.save_summary_steps == 0):
                tensors = {}
                for tensor_key in train_tensor_keys:
                    if tensor_key in batch.keys():
                        tensors[tensor_key] = batch[tensor_key].squeeze().numpy()
                    elif tensor_key.endswith("hat"):
                        tensor_key = tensor_key.split("_")[0]
                        if tensor_key in output_batch.keys():
                            tensors[tensor_key+"_hat"] = output_batch[tensor_key].detach().cpu().squeeze().numpy()
                    else:
                        assert False, f"key not found: {tensor_key}"
                # print(tensors)
                df = pd.DataFrame.from_dict(tensors, orient='columns')
                df["epoch"] = epoch

                with open(os.path.join(logdir, 'train-tensors.csv'), 'a') as f:
                    df[["epoch"]+train_tensor_keys].to_csv(f, header=False)

            # update the average loss
            loss_avg.update(loss.item())

            progress_bar.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            progress_bar.update()

    # visualize gradients
    if epoch % params.save_summary_steps == 0 and args.monitor_grads:
        abs_gradients = {}
        for name, param in model.named_parameters():
            try: # patch here, there were names / params that were 'none'
                abs_gradients[name] = np.abs(param.grad.cpu().numpy()).mean()
                writer.add_histogram("grad-"+name, param.grad, epoch)
                writer.add_scalars("mean-abs-gradients", abs_gradients, epoch)
            except:
                pass
        if params.use_mi:
            for mi, mine in mines.items():
                for name, param in mine.named_parameters():
                    writer.add_histogram("grad-mine-"+mi+"-"+name, param.grad, epoch)

    # compute mean of all metrics in summary
    metrics_mean = {metric:np.nanmean([x[metric] for x in summ]) for metric in summ[0]}
    
    # collect tensors
    Xhat = torch.cat(Xhats,0).view(-1,1)
    Yhat = torch.cat(Yhats,0).view(-1,1)
    Zhat = torch.cat(Zhats,0)
    t    = torch.cat(ts,0)
    X    = torch.cat(Xs,0)
    Xtrue= torch.cat(Xtrues,0)
    Y    = torch.cat(Ys,0)
    
    if params.do_least_squares:
        # after the minibatches, do a single OLS on the whole data
        Zi = torch.cat([torch.ones_like(t), Zhat], 1)
        # add treatment info
        Zt = torch.cat([Zi, t], 1)
        # add x for biased version
        XZt = torch.cat([torch.ones_like(t), Xhat, Zhat, t], 1)

        betas_y_bias       = net.cholesky_least_squares(XZt, Y, intercept=False)
        betas_y_causal     = net.cholesky_least_squares(Zt, Y, intercept=False)
        model.betas_bias   = betas_y_bias
        model.betas_causal = betas_y_causal
        metrics_mean["regr_bias_coef_t"]   = betas_y_bias.squeeze()[-1]
        metrics_mean["regr_bias_coef_z"]   = betas_y_bias.squeeze()[-2]
        metrics_mean["regr_causal_coef_t"] = betas_y_causal.squeeze()[-1]
        metrics_mean["regr_causal_coef_z"] = betas_y_causal.squeeze()[-2]
       
    # create some plots
    xx_scatter    = net.make_scatter_plot(X.numpy(), Xhat.numpy(), xlabel='x', ylabel='xhat') 
    xtruex_scatter= net.make_scatter_plot(Xtrue.numpy(), Xhat.numpy(), xlabel='xtrue', ylabel='xhat') 
    xyhat_scatter = net.make_scatter_plot(X.numpy(), Yhat.numpy(), c=t.numpy(), xlabel='x', ylabel='yhat')
    yy_scatter    = net.make_scatter_plot(Y.numpy(), Yhat.numpy(), c=t.numpy(), xlabel='y', ylabel='yhat') 
    writer.add_figure('x-xhat/train', xx_scatter, epoch+1)
    writer.add_figure('xtrue-xhat/train', xtruex_scatter, epoch+1)
    writer.add_figure('x-yhat/train', xyhat_scatter, epoch+1)
    writer.add_figure('y-yhat/train', yy_scatter, epoch+1)


    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)

    return metrics_mean


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_fn, metrics, params, setting, args,
                       writer=None, logdir=None, restore_file=None, 
                       mines=None, optims_mine=None):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using mnisthe output and labels of each batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from (withoutmnistits extension .pth.tar)
        covar_mode: (bool) does the data-loader give back covariates / additional data
    """

    if params.use_mi:
        # for (mi_name, mi_estimator), mi_optim in zip(mines.items(), optims_mine.values()):
        #     print(mi_name)
        #     print(mi_estimator)
        #     print(mi_optim)
        train_mines       = mines
        train_optims_mine = optims_mine
        val_mines         = train_mines
        val_optims_mine   = None

        # train_mines       = {k: v for k, v in mines.items() if k in ['bnx_x', 'bnz_x']}
        # train_optims_mine = {k: v for k, v in optims_mine.items() if k in ['bnx_x', 'bnz_x']}
        #val_mines         = {k: v for k, v in mines.items() if not (k in ['bnx_x', 'bnz_x'])}
        #val_optims_mine   = {k: v for k, v in optims_mine.items() if not (k in ['bnx_x', 'bnz_x'])}
        # if params.bn_loss_type == 'mi':
        #     train_mines['bnz_x'] = mines['bnz_x']
        #     train_optims_mine['bnz_x'] = mines['bnz_x']
    else:
        train_mines = train_optims_mine = val_mines = val_optims_mine = None

    # setup directories for data
    setting_home = setting.home
    if not args.fase == "feature":
        data_dir = os.path.join(setting_home, "data")
    else:
        if setting.mode3d:
            data_dir = "data"
        else:
            data_dir = "slices"
    covar_mode = setting.covar_mode

    x_frozen = False


    best_val_metric = 0.0
    if "loss" in setting.metrics[0]:
        best_val_metric = 1.0e6

    val_preds = np.zeros((len(val_dataloader.dataset), params.num_epochs))

    for epoch in range(params.num_epochs):

        # Run one epoch
        logging.info(f"Epoch {epoch+1}/{params.num_epochs}; setting: {args.setting}, fase {args.fase}, experiment: {args.experiment}")

        # stop gradients for freezing x part of model
        if (params.freeze_x and (epoch > params.bn_loss_lag_epochs)) and (not x_frozen):
            x_frozen = True    # flag for not doing this twice
            params.alpha = 0.0 # no more loss to x
            print(f"freezing layers before x")
            pre_x_layers = ["encoder", "fcs", "fcx", "fcx2", "regressor_x"]
            if params.bn_place == "single-regressor":
                # freeze everything except for last regressor
                pre_x_layers.append("fcy")
            keep_grad_layers = []
            for name, param in model.named_parameters():
                for x_layer in pre_x_layers:
                    if x_layer in name:
                        print(f"turning off gradient for {name}")
                        param.requires_grad_(False)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print(f"keeping gradient for {name}")

        # compute number of batches in one epoch (one full pass over the training set)
        train_metrics = train(model, optimizer, loss_fn, train_dataloader, metrics, params, setting, writer, epoch, 
                              mines=train_mines, optims_mine=train_optims_mine)

        for metric_name in train_metrics.keys():
            metric_vals = {'train': train_metrics[metric_name]}
            writer.add_scalars(metric_name, metric_vals, epoch+1)


        # for name, param in model.named_parameters():
        #     writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch+1)
        
        if epoch % params.save_summary_steps == 0:

            # Evaluate for one epoch on validation set
            valid_metrics, outtensors = evaluate(model, loss_fn, val_dataloader, metrics, params, setting, epoch, writer, 
                                            mines=val_mines, optims_mine=val_optims_mine)
            valid_metrics["intercept"] = model.regressor.fc.bias.detach().cpu().numpy()
            
            
            for name, module in model.regressor.named_children():
                if name == "t":
                    valid_metrics["b_t"] = module.weight.detach().cpu().numpy()
                elif name == "zt":
                    weights = module.weight.detach().cpu().squeeze().numpy().reshape(-1)
                    for i, weight in enumerate(weights):
                        valid_metrics["b_zt"+str(i)] = weight
                else:
                    pass
            for metric_name in valid_metrics.keys():
                metric_vals = {'valid': valid_metrics[metric_name]}
                writer.add_scalars(metric_name, metric_vals, epoch+1)

            # create plots
            val_df = val_dataloader.dataset.df
            xx_scatter    = net.make_scatter_plot(val_df.x.values, outtensors['xhat'], xlabel='x', ylabel='xhat') 
            xtruex_scatter= net.make_scatter_plot(val_df.x_true.values, outtensors['xhat'], xlabel='x', ylabel='xhat') 
            xyhat_scatter = net.make_scatter_plot(val_df.x.values, outtensors['predictions'], c=val_df.t, xlabel='x', ylabel='yhat')
            zyhat_scatter = net.make_scatter_plot(val_df.z.values, outtensors['predictions'], c=val_df.t, xlabel='z', ylabel='yhat')
            yy_scatter    = net.make_scatter_plot(val_df.y.values, outtensors['predictions'], c=val_df.t, xlabel='yhat', ylabel='y') 
            writer.add_figure('x-xhat/valid', xx_scatter, epoch+1)
            writer.add_figure('xtrue-xhat/valid', xtruex_scatter, epoch+1)
            writer.add_figure('x-yhat/valid', xyhat_scatter, epoch+1)
            writer.add_figure('z-yhat/valid', zyhat_scatter, epoch+1)
            writer.add_figure('y-yhat/valid', yy_scatter, epoch+1)

            if args.visdom and ("b_t" in valid_metrics.keys()) and ("loss_y" in valid_metrics.keys()):
                
                # visualize loss vs ate
                try:
                    ate_loss_plt
                except NameError:
                    ate_loss_plt = viz.line(
                        X = valid_metrics["b_t"].reshape(1,1),
                        Y = valid_metrics["loss_y"].reshape(1,1),
                        opts=dict(
                            # legend=logdir,
                            xtickmin=-0.5,
                            xtickmax=1.5,
                            xtickstep=0.5,
                            ytickmin=0.0,
                            ytickmax=3.0,
                            ytickstep=0.5,
                            title="b_t vs loss",
                            xlabel="b_t", ylabel="loss_y" 
                            # linecolor=np.array([255, 0, 0]).reshape(1,3)
                        ),
                        name=args.experiment
                    )
                else:
                    viz.line(
                        X = valid_metrics["b_t"].reshape(1,1),
                        Y = valid_metrics["loss_y"].reshape(1,1),
                        win=ate_loss_plt,
                        # opts=dict(
                        #     linecolor=np.array([255, 0, 0]).reshape(1,3) + np.array([0, 255, 0]).reshape(1,3) * epoch>params.bn_loss_lag_epochs
                        # ),
                        name=args.experiment,
                        update='append'
                    )


        # writer.add_scalars('valid', valid_metrics, epoch+1)
        # writer.add_scalars('train', train_metrics, epoch+1)
        # metric_dict = {m: {'train': train_metrics[m], 'valid': valid_metrics[m]} for m in train_metrics.keys()}


            if params.save_preds:
                # writer.add_histogram("predictions", preds)
                if setting.num_classes == 1:
                    val_preds[:, epoch] = np.squeeze(outtensors['predictions'])
                    
                    # write preds to file
                    pred_fname = os.path.join(setting.home, setting.fase+"-fase", "preds_val.csv")
                    with open(pred_fname, 'ab') as f:
                        np.savetxt(f, preds.T, newline="")

                np.save(os.path.join(setting.home, setting.fase+"-fase", "preds.npy"), preds)

            #if params.multi_task and params.use_mi:
            #    val_metric = valid_metrics["bnx_x"]
            else:
                val_metric = valid_metrics[setting.metrics[0]]
            # val_acc = valid_metrics['accuracy']
            if "loss" in str(setting.metrics[0]):
                is_best = val_metric<=best_val_metric
                # print("new best, old: {:.3f}, new: {:.3f}".format(best_val_metric, val_metric))
            else:
                is_best = val_metric>=best_val_metric

            # Save weights
            state_dict = model.state_dict()
            optim_dict = optimizer.state_dict()

            # exclude weights from layers that get changed
            # exclude_layers = ["bottleneck", "bnbn", "fc2"]
            # state_dict = {s: state_dict[s] for s in state_dict.keys() if s.split(".")[0] not in exclude_layers}
            # optim_dict = {s: optim_dict[s] for s in optim_dict.keys() if s.split(".")[0] not in exclude_layers}

            state = {
                'epoch': epoch+1,
                'state_dict': state_dict,
                'optim_dict': optim_dict
            }

            if params.use_mi:
                for mi_name, mine in mines.items():
                    state[mi_name] = mine.state_dict()
                    state[mi_name+"_optim"] = optims_mine[mi_name].state_dict()

            utils.save_checkpoint(state,
                                is_best=is_best,
                                checkpoint=logdir)

            # If best_eval, best_save_path
            valid_metrics["epoch"] = epoch
            if is_best:
                logging.info("- Found new best {}: {:.3f}".format(setting.metrics[0], val_metric))
                best_val_metric = val_metric

                # Save best val metrics in a json file in the model directory
                best_json_path = os.path.join(logdir, "metrics_val_best_weights.json")
                utils.save_dict_to_json(valid_metrics, best_json_path)

            # Save latest val metrics in a json file in the model directory
            last_json_path = os.path.join(logdir, "metrics_val_last_weights.json")
            utils.save_dict_to_json(valid_metrics, last_json_path)
    
    # final evaluation
    writer.export_scalars_to_json(os.path.join(logdir, "all_scalars.json"))

    if args.save_preds:
        np.save(os.path.join(setting.home, setting.fase + "-fase", "val_preds.npy"), val_preds)

    # if setting.covar_mode or setting.num_classes == 1:
    #     evaluate_predictions(os.path.join(setting.home, "data"),
    #                          os.path.join(setting.home, setting.fase + "-fase"))


if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()


    # Load information from last setting if none provided:
    last_defaults = utils.Params("last-defaults.json")
    if args.setting == "":
        print("using last default setting")
        args.setting = last_defaults.dict["setting"]
        for param, value in last_defaults.dict.items():
            print("{}: {}".format(param, value))
    else:
        with open("last-defaults.json", "r+") as jsonFile:
            defaults = json.load(jsonFile)
            tmp = defaults["setting"]
            defaults["setting"] = args.setting
            jsonFile.seek(0)  # rewind
            json.dump(defaults, jsonFile)
            jsonFile.truncate()

    # setup visdom environment
    if args.visdom:
        from visdom import Visdom
        viz = Visdom(env=f"lidcr_{args.setting}_{args.fase}_{args.experiment}")

    # load setting (data generation, regression model etc)
    setting_home = os.path.join(args.setting_dir, args.setting)
    setting = utils.Params(os.path.join(setting_home, "setting.json"))
    setting.home = setting_home

    # when not specified in call, grab model specification from setting file
    if setting.cnn_model == "":
        json_path = os.path.join(args.model_dir, "t-suppression", args.experiment+".json")
    else:
        json_path = os.path.join(args.model_dir, setting.cnn_model, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    if not os.path.exists(os.path.join(setting.home, args.fase + "-fase")):
        os.makedirs(os.path.join(setting.home, args.fase + "-fase"))
    shutil.copy(json_path, os.path.join(setting_home, args.fase + "-fase", "params.json"))
    params = utils.Params(json_path)
    # covar_mode = setting.covar_mode
    # mode3d = setting.mode3d
    parallel = args.parallel

    params.device = None
    if not args.disable_cuda and torch.cuda.is_available():
        params.device = torch.device('cuda')
        params.cuda = True
        # switch gpus for better use when running multiple experiments
        if not args.parallel:
            if os.path.exists("last-cuda-0.flag"):
                torch.cuda.set_device(1)
                shutil.move("last-cuda-0.flag", "last-cuda-1.flag")
            else:
                torch.cuda.set_device(0)
                shutil.move("last-cuda-1.flag", "last-cuda-0.flag")
            # last_defaults = utils.Params("last-defaults.json")
            # last_cuda_id = last_defaults.cuda_id
            # new_cuda_id = int((last_cuda_id + 1) % 2)
            # torch.cuda.set_device(new_cuda_id)
            # new_defaults = last_defaults
            # new_defaults.cuda_id = new_cuda_id
            # utils.save_dict_to_json(new_defaults, "last-defaults.json")        
    else:
        params.device = torch.device('cpu')

    # adapt fase
    setting.fase = args.fase
    if args.fase == "x":
        # setting.outcome = ["x"]
        setting.num_classes = 1
        setting.counterfactuals = False
        setting.covar_mode = False
        # setting.metrics = setting.metrics + ["total_loss"]
        # setting.metrics = [x for x in setting.metrics if x != "total_loss"]
        params.bottleneck_loss = True
        params.bn_loss_lag_epochs = params.num_epochs
        # setting.metrics = setting.metrics + ["bottleneck_loss"]
    elif args.fase == "y":
        pass
        # setting.outcome = ["y"]
        # setting.covar_mode = False
        # setting.metrics = setting.metrics + ["bottleneck_loss"]
    elif args.fase == "yt":
        # setting.metrics = setting.metrics + ["bottleneck_loss"]
        setting.covar_mode = True
        # setting.outcome = ["y"]
        # params.suppress_t_epochs = 0
        # if args.cold_start: 
        #     params.suppress_t_epochs = 90
        # params.num_epochs = 180
    elif args.fase == "xy":
        # setting.outcome = ["x", "y"]
        setting.covar_mode = True
        # setting.metrics = setting.metrics + ["bottleneck_loss"]
        setting.metrics = [x for x in setting.metrics if x!="total_loss"]
        # params.num_epochs = 20
        # params.bottleneck_loss = False
        # params.suppress_t_epochs = 0
    elif args.fase == "xybn":
        # setting.outcome = ["x", "y"]
        # setting.covar_mode = True
        # setting.metrics = setting.metrics + ["bottleneck_loss", "ate"]
        # setting.metrics = [x for x in setting.metrics if x!="total_loss"]
        # params.suppress_t_epochs = 10
        params.bottleneck_loss = True
    elif args.fase == "bn":
        setting.outcome = ["x", "y"]
        setting.covar_mode = True
        params.bottleneck_loss = True
        # setting.metrics = setting.metrics + ["bottleneck_loss"]
    elif args.fase == "z":
        setting.outcome = ["z"]



    setting.metrics = pd.Series(setting.metrics).drop_duplicates().tolist()
    print("metrics {}:".format(setting.metrics))

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda: torch.cuda.manual_seed(230)

    # Set the logger
    logdir=os.path.join(setting_home, setting.fase+"-fase", "runs")
    if not args.experiment == '':
        logdir=os.path.join(logdir, args.experiment)
    if not os.path.isdir(logdir):
        os.makedirs(logdir)

    # copy params as backupt to logdir
    shutil.copy(json_path, os.path.join(logdir, "params.json"))

    # utils.set_logger(os.path.join(args.model_dir, 'train.log'))
    utils.set_logger(os.path.join(logdir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # fetch dataloaders
    # dataloaders = data_loader.fetch_dataloader(['train', 'val'], args.data_dir, params)
    # dataloaders = data_loader.fetch_dataloader(['train', 'val'], "data", params)
    # dataloaders = data_loader.fetch_dataloader(types = ["train", "valid"])
    dataloaders = data_loader.fetch_dataloader(args, params, setting, ["train", "valid"])
    train_dl = dataloaders['train']
    valid_dl = dataloaders['valid']

    if setting.num_classes > 1 and params.balance_classes:
        train_labels = train_dl.dataset.df[setting.outcome[0]].values
        class_weights = compute_class_weight('balanced', np.unique(train_labels), train_labels)
    # valid_dl = train_dl

    logging.info("- done.")

    # print("allocated mem after dataloaders %s" % (torch.cuda.max_memory_allocated(0)))

    # Define the model and optimizer
    # if covar_mode:
        # model = net.TNet(params).cuda() if params.cuda else net.TNet(params)
    # else:
    # model = net.Net(params).to(params.device, non_blocking=True)

    if args.intercept:
        assert len(setting.outcome) == 1, "Multiple outcomes not implemented for intercept yet"
        print("running intercept mode")
        mu = valid_dl.dataset.df[setting.outcome].values.mean()
        def new_forward(self, x, data, mu=mu):
            intercept = torch.autograd.Variable(mu * torch.ones((x.shape[0],1)), requires_grad=False).to(params.device, non_blocking=True)
            bn_activations = torch.autograd.Variable(torch.zeros((x.shape[0],)), requires_grad=False).to(params.device, non_blocking=True)
            return {setting.outcome[0]: intercept, "bn": bn_activations}

        net.Net3D.forward = new_forward
        params.num_epochs = 1
        setting.metrics = []
        logdir = os.path.join(logdir, "intercept")

    if setting.mode3d:
        model = net.Net3D(params, setting).to(params.device)
    else:
        model = net.CausalNet(params, setting).to(params.device)

    optimizers = {'sgd': optim.SGD, 'adam': optim.Adam}

    if params.use_mi:
        if params.mi_estimator == 'mine':
            MI_ESTIMATOR = net.MINE
        elif params.mi_estimator == 'jsd':
            MI_ESTIMATOR = net.JSD
        else:
            raise NotImplementedError(f'MI estimator not implemented: {params.mi_estimator}, should be mine or jsd')
        # prepare mutual information estimators
        mines = dict(
            # these are part of the loss function
            bnx_x = MI_ESTIMATOR(params.regressor_x_dim+1).to(params.device), # MI from activation(s) that should represent x to real x
            bnz_bnx = MI_ESTIMATOR(params.regressor_x_dim+params.regressor_z_dim).to(params.device),
            
            monitor_bnx_x = MI_ESTIMATOR(params.regressor_x_dim+1).to(params.device), # MI from activation(s) that should represent x to real x
            monitor_bnz_bnx = MI_ESTIMATOR(params.regressor_x_dim+params.regressor_z_dim).to(params.device),

            # add more for monitoring purposes
            bnz_x = MI_ESTIMATOR(params.regressor_z_dim+1).to(params.device), # MI from other activations to measured x
            bnz_z = MI_ESTIMATOR(params.regressor_z_dim+1).to(params.device)  # MI from other activations to true z
        )
        # each MI estimator gets their own optimizer
        optims_mine = {}
        for mi_name, mi_estimator in mines.items():
            optims_mine[mi_name] = optim.Adam(mi_estimator.parameters(), lr=params.mi_lr)
    else:
        mines = optims_mine = None
        
    if parallel:
        print("parallel mode")
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    if params.momentum > 0:
        optimizer = optimizers[params.optimizer](model.parameters(), lr=params.learning_rate, weight_decay=params.wd, momentum=params.momentum)
    else:
        optimizer = optimizers[params.optimizer](model.parameters(), lr=params.learning_rate, weight_decay=params.wd)

    # if params.use_mi:
    #     optimizer.add_param_group({'params': mine.parameters()})

    if setting.covar_mode and params.lr_t_factor != 1:
        optimizer = net.speedup_t(model, params)

    if args.restore_last and (not args.cold_start):
        print("Loading state dict from last running setting")
        utils.load_checkpoint(os.path.join(setting.home, args.fase + "-fase", "last.pth.tar"), model, strict=False)
    elif args.restore_warm:
        utils.load_checkpoint(os.path.join(setting.home, 'warm-start.pth.tar'), model, strict=False)
    else:
        if not setting.fase in ["x", "feature"]:
            if args.cold_start:
                print("warning, not loading weights from previous fase, since cold-start = True")
            else:
                if (args.restore_file is not None) and (args.fase == "xybn"):
                    restore_path = os.path.join(setting.home, "xy-fase", args.restore_file + '.pth.tar')
                    logging.info("Restoring parameters from {}".format(restore_path))
                    if params.use_mi:
                        # utils.load_checkpoint(restore_path, model, mines=mines, optims_mine=optims_mine)
                        utils.load_checkpoint(restore_path, model, mines=mines)
                    else:
                        utils.load_checkpoint(restore_path, model)
                else:
                    if args.use_last:
                        state_dict_type = "last"
                    else:
                        state_dict_type = "best"

                    if setting.fase == "y":
                        print("loading state-dict from x-fase")
                        utils.load_checkpoint(os.path.join(setting.home, "x-fase", state_dict_type+".pth.tar"), model, strict=False)
                    elif setting.fase == "yt":
                        # print("loading state-dict from y-fase")
                        # utils.load_checkpoint(os.path.join(setting.home, "y-fase", state_dict_type+".pth.tar"), model, strict=False)
                        print("loading state-dict from y-fase")
                        utils.load_checkpoint(os.path.join(setting.home, "y-fase", state_dict_type+".pth.tar"), model, strict=False)
                    elif setting.fase == "xy":
                        print("loading state-dict from yt-fase")
                        utils.load_checkpoint(os.path.join(setting.home, "yt-fase", state_dict_type+".pth.tar"), model, strict=False)
                    elif setting.fase == "xybn":
                        print("loading state-dict from xy-fase: last")
                        # utils.load_checkpoint(os.path.join(setting.home, "xy-fase", "last.pth.tar"), model, strict=False)
                        utils.load_checkpoint(os.path.join(setting.home, "xy-fase", state_dict_type+".pth.tar"), model, mines=mines, optims_mine = optims_mine, strict=False)
                        # utils.load_checkpoint(os.path.join(setting.home, "xybn-fase", "last.pth.tar"), model, strict=False)
                        # optimizer = net.softfreeze_conv_layers(model, params)
                    elif setting.fase == "bn":
                        print("loading state-dict from xy-fase")
                        utils.load_checkpoint(os.path.join(setting.home, "xy-fase", state_dict_type+".pth.tar"), model, strict=False)
                        # optimizer = net.softfreeze_conv_layers(model, params)
                    else:
                        assert False, "invalid fase: {}, should be in {x, y, xy, bn}".format(setting.fase)
    
    # if setting.fase == "bn":
        # net.freeze_conv_layers(model)

    # fetch loss function and metrics
    if setting.num_classes > 1 and params.balance_classes:
        loss_fn = net.get_loss_fn(setting, weights=class_weights)
    else:
        loss_fn = net.get_loss_fn(setting)
    # metrics = {metric:net.all_metrics[metric] for metric in setting.metrics}
    metrics = None

    if params.monitor_train_tensors:
        print(f"Recording all train tensors")
        import csv   
        train_tensor_keys = ['t','x', 'z', 'y', 'x_hat', 'z_hat', 'y_hat']
        with open(os.path.join(logdir, 'train-tensors.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch']+train_tensor_keys)

    # Train the model
    # print(model)
    # print(summary(model, (3, 224, 224), batch_size=1))
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    for split, dl in dataloaders.items():
        logging.info("Number of %s samples: %s" % (split, str(len(dl.dataset))))
        # logging.info("Number of valid examples: {}".format(len(valid.dataset)))

    
    with SummaryWriter(logdir) as writer:
        # train(model, optimizer, loss_fn, train_dl, metrics, params)
        train_and_evaluate(model, train_dl, valid_dl, optimizer, loss_fn, metrics, params, setting, args,
                           writer, logdir, args.restore_file, 
                           mines, optims_mine)
