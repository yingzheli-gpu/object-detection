from ultralytics import YOLO

from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.models.yolo import classify, detect, segment
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils import yaml_load, LOGGER, RANK, TQDM,colorstr,checks
from ultralytics.utils.checks import check_yaml
import torch
import numpy as np
import gc
import time,math
import warnings
from torch import distributed as dist
from ultralytics.utils.torch_utils import (
    autocast,
)
from torch import nn
import torch.optim as optim
from ultralytics.utils.transform import MultiScaleDeformableAttentionTransformer as MSDATrans
device = 'cuda'
def compute_loss(func,yp,yt):
    if isinstance(yp,list):
        loss=func(yp[0],yt[0])
        for i in range(1,len(yp)):
            loss+=func(yp[i],yt[i])
        return loss
    return func(yp,yt)
msd_teacher = MSDATrans(base_channels=64, num_inputs=3, num_offsets=4).to(device)
optimizer_msd = optim.Adam(msd_teacher.parameters(), lr=1e-2, betas=(0.9, 0.999))
lr_rate=10
distill_loss_func = nn.MSELoss(reduction="mean")
diff_loss_func = nn.MSELoss(reduction="mean")

def compute_distill_loss(students,teachers):
    global distill_loss_func, diff_loss_func
    msd_teacher = MSDATrans(base_channels=64, num_inputs=3, num_offsets=4).to(device)
    f_trans = msd_teacher(students)  # 经过 MSDATrans 的特征变换
    diff_loss, dist_loss = 0, 0
    for i in range(len(f_trans)):
        diff_loss += 1 / (diff_loss_func(f_trans[i].detach(), students[i]) + 1e-6)
        dist_loss += distill_loss_func(students[i], f_trans[i].detach())
    dist_loss /= len(f_trans)
    bs = students[0].shape[0]
    #total_loss = (diff_loss + dist_loss) / bs * lr_rate  # 统一缩放损失
    total_loss = (diff_loss + dist_loss) / 2  # 统一缩放损失
    return total_loss

activation = {}
def get_activation(name):
    global activation
    def hook(model, inputs, outputs):
        activation[name] = outputs
    return hook
def get_hooks(distill_ids,student_model,teacher_model):
    hooks = []
    for i,idx in enumerate(distill_ids):
        # S-model
        hooks.append(student_model.model.model[idx].register_forward_hook(get_activation(f"s_f{i}")))
        # T-model
        hooks.append(teacher_model.model.model[idx].register_forward_hook(get_activation(f"t_f{i}")))
    return hooks

class Converter(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super(Converter, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, k // 2, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU6(inplace=True) if act else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

def _do_train_distill(self: BaseTrainer, world_size,teacher_model,
    distill_ids,ts_c_msg
):
    if True:
        #===========设置需要进行剪枝的层==============
        global activation
        if True:
            #device='cuda'
            # feature convert
            #添加特征转换层，使蒸馏时student的chanel与teacher输出的chanel一致    
            S_Converters=[]
            T_Converters=[]
            for c_ms in ts_c_msg:
                if c_ms[0]!=c_ms[1]:
                    S_Converter_ = Converter(c_ms[0], c_ms[1], act=True).to(device).train()
                    T_Converter_ = nn.ReLU6().to(device).train()
                else:
                    S_Converter_ = nn.Identity()
                    T_Converter_ = nn.Identity()
                S_Converters.append(S_Converter_)
                T_Converters.append(T_Converter_)

        """Train completed, evaluate and plot if specified by arguments."""
        if world_size > 1:
            self._setup_ddp(world_size)
        self._setup_train(world_size)

        nb = len(self.train_loader)  # number of batches
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # warmup iterations
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")
        LOGGER.info(
            f'Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n'
            f'Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n'
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f'Starting training for ' + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
        )
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        epoch = self.start_epoch
        self.optimizer.zero_grad()  # zero any resumed gradients to ensure stability on train start
        while True:
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            #----插入hook
            hooks=get_hooks(distill_ids,self,teacher_model)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                self.scheduler.step()

            self.model.train()
            msd_teacher.train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            # Update dataloader attributes (optional)
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()

            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            self.tloss = None
            distill_loss_list=[]
            for i, batch in pbar:
                self.run_callbacks("on_train_batch_start")
                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    for j, x in enumerate(self.optimizer.param_groups):
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x["lr"] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                # Forward
                with autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    # batch['im_file'] str list, 图片路径
                    # batch['img'] tensor 输入图像
                    self.loss, self.loss_items = self.model(batch)

                    #----基于hook，获取两个模型的中间层
                    with torch.no_grad():
                        t_pred = teacher_model.model(batch['img'])
                    #----#基于Converters，进行通道对齐
                    s_fs,t_fs=[],[]
                    for i,idx in enumerate(distill_ids):
                        if isinstance(activation[f"s_f{i}"],list):
                            #这里为输出层，不需要进行对齐，本来就是一样的shape
                            s_f_ = activation[f"s_f{i}"]#
                            t_f_ = activation[f"t_f{i}"]
                        else:
                            s_f_ = S_Converters[i](activation[f"s_f{i}"]) #基于Converters，对某一layer进行通道对齐
                            t_f_ = T_Converters[i](activation[f"t_f{i}"]) 
                        #---相同的shape，不需要 T_Converters
                        # s_f_ = activation[f"s_f{i}"]#
                        # t_f_ = activation[f"t_f{i}"]
                        s_fs.append(s_f_)
                        t_fs.append(t_f_)
                    #----#计算中间特征的loss计算
                    #distill_loss=compute_distill_loss(s_fs,t_fs)
                    #distill_loss_list.append(distill_loss.cpu().item())
                    if i % 5 == 0:
                       for param in self.model.parameters():
                           param.requires_grad = False
                       for param in msd_teacher.parameters():
                           param.requires_grad = True

                       f_trans = msd_teacher(s_fs)
                       tea_loss = sum(1 / (diff_loss_func(f_trans[j], s_fs[j]) + 1e-6) for j in range(len(f_trans)))

                       optimizer_msd.zero_grad()
                       tea_loss.backward()
                       optimizer_msd.step()

                    for param in self.model.parameters():
                        param.requires_grad = True
                    for param in msd_teacher.parameters():
                        param.requires_grad = False
                    f_trans = msd_teacher(s_fs)
                    diff_loss, dist_loss = 0, 0
                    for i in range(len(f_trans)):
                        #diff_loss += 1 / (diff_loss_func(f_trans[i].detach(), s_fs[i]) + 1e-6)
                        dist_loss += distill_loss_func(s_fs[i], f_trans[i].detach())
                    dist_loss /= len(f_trans)
                    distill_loss=dist_loss
                    distill_loss_list.append(distill_loss.cpu().item())
                    #----将蒸馏loss添加到全局loss中
                    self.loss+= distill_loss

                    if RANK != -1:
                        self.loss *= world_size
                    self.tloss = (
                        (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items
                    )

                # Backward
                self.scaler.scale(self.loss).backward()

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                    # Timed stopping
                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                        if RANK != -1:  # if DDP training
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                            self.stop = broadcast_list[0]
                        if self.stop:  # training time exceeded
                            break

                # Log
                mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
                loss_len = self.tloss.shape[0] if len(self.tloss.shape) else 1
                losses = self.tloss if loss_len > 1 else torch.unsqueeze(self.tloss, 0)
                if RANK in {-1, 0}:
                    distill_loss_mean=sum(distill_loss_list)/len(distill_loss_list)
                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * (2 + loss_len+1))
                        % (f"{epoch + 1}/{self.epochs}", mem, *losses,distill_loss_mean, batch["cls"].shape[0], batch["img"].shape[-1])
                    )
                    self.run_callbacks("on_batch_end")
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks("on_train_batch_end")

            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers
            for hook in hooks:
                hook.remove()
            self.run_callbacks("on_train_epoch_end")
            if RANK in {-1, 0}:
                final_epoch = epoch + 1 >= self.epochs
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

                # Validation
                if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                    self.metrics, self.fitness = self.validate()
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

                # Save model
                if self.args.save or final_epoch:
                    self.save_model()
                    self.run_callbacks("on_model_save")

            # Scheduler
            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            if self.args.time:
                mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                self._setup_scheduler()
                self.scheduler.last_epoch = self.epoch  # do not move
                self.stop |= epoch >= self.epochs  # stop if exceeded epochs
            self.run_callbacks("on_fit_epoch_end")
            gc.collect()
            torch.cuda.empty_cache()  # clear GPU memory at end of epoch, may help reduce CUDA out of memory errors

            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks
            epoch += 1

        if RANK in {-1, 0}:
            # Do final val with best.pt
            LOGGER.info(
                f"\n{epoch - self.start_epoch + 1} epochs completed in "
                f"{(time.time() - self.train_time_start) / 3600:.3f} hours."
            )
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        gc.collect()
        torch.cuda.empty_cache()
        self.run_callbacks("teardown")

from ultralytics.utils.dist import ddp_cleanup, generate_ddp_command
import subprocess,os
def train_distill(self: BaseTrainer,teacher_model,distill_ids,ts_c_msg
):
    if True:
        """Allow device='', device=None on Multi-GPU systems to default to device=0."""
        if isinstance(self.args.device, str) and len(self.args.device):  # i.e. device='0' or device='0,1,2,3'
            world_size = len(self.args.device.split(","))
        elif isinstance(self.args.device, (tuple, list)):  # i.e. device=[0, 1, 2, 3] (multi-GPU from CLI is list)
            world_size = len(self.args.device)
        elif self.args.device in {"cpu", "mps"}:  # i.e. device='cpu' or 'mps'
            world_size = 0
        elif torch.cuda.is_available():  # i.e. device=None or device='' or device=number
            world_size = 1  # default to device 0
        else:  # i.e. device=None or device=''
            world_size = 0

        # Run subprocess if DDP training, else train normally
        if world_size > 1 and "LOCAL_RANK" not in os.environ:
            # Argument checks
            if self.args.rect:
                LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with Multi-GPU training, setting 'rect=False'")
                self.args.rect = False
            if self.args.batch < 1.0:
                LOGGER.warning(
                    "WARNING ⚠️ 'batch<1' for AutoBatch is incompatible with Multi-GPU training, setting "
                    "default 'batch=16'"
                )
                self.args.batch = 16

            # Command
            cmd, file = generate_ddp_command(world_size, self)
            try:
                LOGGER.info(f'{colorstr("DDP:")} debug command {" ".join(cmd)}')
                subprocess.run(cmd, check=True)
            except Exception as e:
                raise e
            finally:
                ddp_cleanup(self, str(file))
                

        else:
            self._do_train_distill(world_size,teacher_model,distill_ids,ts_c_msg)

#修改训练时的loss项目
def progress_string(self: BaseTrainer):
    """Returns a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
    return ("\n" + "%11s" * (4 + len(self.loss_names)+1)) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            'distill-loss',
            "Instances",
            "Size",
        )
from ultralytics.engine.trainer import BaseTrainer
def train_v2_distill(self: YOLO, teacher_model,distill_ids,ts_c_msg,trainer=None,**kwargs):
    self._check_is_pytorch_model()
    if self.session:  # Ultralytics HUB session
        if any(kwargs):
            LOGGER.warning('WARNING ⚠️ using HUB training arguments, ignoring local training arguments.')
        kwargs = self.session.train_args
    overrides = self.overrides.copy()
    overrides.update(kwargs)
    if kwargs.get('cfg'):
        LOGGER.info(f"cfg file passed. Overriding default params with {kwargs['cfg']}.")
        overrides = yaml_load(check_yaml(kwargs['cfg']))
    overrides['mode'] = 'train'
    if not overrides.get('data'):
        raise AttributeError("Dataset required but missing, i.e. pass 'data=coco128.yaml'")
    if overrides.get('resume'):
        overrides['resume'] = self.ckpt_path

    self.task = overrides.get('task') or self.task
    self.trainer = detect.DetectionTrainer(overrides=overrides, _callbacks=self.callbacks)
    #TASK_MAP[self.task][1](overrides=overrides, _callbacks=self.callbacks)
    self.trainer.train = train_distill.__get__(self.trainer)
    self.trainer.__setattr__("_do_train_distill", _do_train_distill.__get__(self.trainer))
    self.trainer.__setattr__("progress_string", progress_string.__get__(self.trainer))
    self.trainer.model = self.model
    self.trainer.hub_session = self.session  # attach optional HUB session
    self.trainer.train(teacher_model,distill_ids,ts_c_msg)
    # Update model and cfg after training
    if RANK in (-1, 0):
        self.model, _ = attempt_load_one_weight(str(self.trainer.best))
        self.overrides = self.model.args
        self.metrics = getattr(self.trainer.validator, 'metrics', None)
