from ultralytics import YOLO
#from feature_distill_utils import train_v2_distill
from FGD import train_v2_distill
import torch
#torch.distributed.init_process_group(backend="nccl")
#local_rank = torch.distributed.get_rank()
#device = torch.device(f"cuda:{local_rank}")

if __name__ == '__main__':
    path= "ultralytics/cfg/models/x/yolov8l.yaml"
    teacher_model=YOLO(path).cuda()

    path= "ultralytics/cfg/models/x/yolov8.yaml"
    #path=r"yolo11n.yaml"
    model=YOLO(path).cuda()
    data= "ultralytics/cfg/datasets/my_data.yaml"
    model.__setattr__("train_v2_distill", train_v2_distill.__get__(model))
    # 如果没有实现自定义loss，user_loss设置为false
    ts_c_msg=[
        (128,128),# student输出128维，teacher输出256维
        (256,256),
        (512,512),
    ]
    model.train_v2_distill(
        teacher_model=teacher_model,distill_ids=[17,20,23],ts_c_msg=ts_c_msg,
        data=data, name='distill_URPC2019_ft_n_0.01',
        lr0=0.001,
        epochs=200,batch=8,workers=8)  # 训练模型
    

    del model
