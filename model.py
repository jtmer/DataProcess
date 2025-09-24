import logging
import os
import sys
import time

import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from torch.utils.data import DataLoader

import matplotlib
import torch.cuda.amp as amp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from transformers import AutoModelForCausalLM

def is_pycharm():
    for key, value in os.environ.items():
        if key == "PYCHARM_HOSTED":
            print(f"PYCHARM_HOSTED={value}")
            return True

matplotlib.use('TkAgg') if is_pycharm() else None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

def is_pycharm():
    for key, value in os.environ.items():
        if key == "PYCHARM_HOSTED":
            print(f"PYCHARM_HOSTED={value}")
            return True


matplotlib.use('TkAgg') if is_pycharm() else None

class TimerXL:
    def __init__(self, model_name, ckpt_path, device):
        self.model_name = model_name
        self.patch_len = 96
        self.device = self.choose_device(device)
        print(f'self.device: {self.device}')
        self.model = AutoModelForCausalLM.from_pretrained(
            #'/data/qiuyunzhong/Training-LTSM/checkpoints/models--thuml--timer-base/snapshots/35a991e1a21f8437c6d784465e87f24f5cc2b395',
            ckpt_path,
            trust_remote_code=True).to(self.device)
    
    def choose_device(self, device):
        if 'cpu' == device:
            return 'cpu'
        elif 'cuda' in device:
            idx = int(device.split(':')[-1])
            os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
            return 'cuda:0'
        else:
            raise ValueError(f'Unknown device: {device}')
    def forcast(self, data, pred_len):
        
        if len(data.shape) == 3:
            data = torch.tensor(data[:,:,-1]).float().to(self.device)
            # print('data.shape=', data.shape)
        pred = self.model.generate(data, max_new_tokens=pred_len)
        pred = pred.unsqueeze(2).detach().to('cpu').numpy()
        return pred

def get_model(model_name, device, args=None):
    if model_name == 'TimerXL':
        model = TimerXL(model_name, '/data/qiuyunzhong/CKPT/models--thuml--timer-base/snapshots/d2a36bed78d84e5699e3f634fc1e5f64bccdb137', device)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    return model

def time_start():
    return time.time()

def log_time_delta(t, event_name):
    d = time.time() - t
    print(f"{event_name} time: {d}")


# # Example usage
# if __name__ == "__main__":
#     seq_len = 96 * 4
#     pred_len = 192
#     # dataset = get_dataset('ETTh1')
#     # dataset = get_dataset('ETTm1')
#     # dataset = get_dataset('Exchange')
#     dataset = get_dataset('Weather')
#     # seq = dataset.np_data_dict['OT'][dataset.train_start:dataset.train_end][:seq_len]
#     # truth_total = dataset.np_data_dict['OT'][dataset.train_start:dataset.train_end][:seq_len + pred_len]

#     mode, target_column, max_seq_len, augmentor, num_sample, batch_size = \
#         'train', 'OT', seq_len, Augmentor('none', 'fix'), 10, 10
#     custom_dataset = CustomDataset(dataset, mode, target_column, max_seq_len, pred_len, augmentor, num_sample)
#     dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=False)
#     idx, history, label = next(iter(dataloader))  # batch, time, feature
#     # history, label = history.numpy(), label.numpy()
#     # 从4维
#     history = history.reshape(batch_size, seq_len, 1)
#     label = label.reshape(batch_size, pred_len, 1)

#     # 对每个 batch 分别计算均值并进行缩放
#     history_transformed = np.zeros_like(history)
#     label_transformed = np.zeros_like(label)
#     scalers = []
#     for i in range(batch_size):
#         scaler = StandardScaler()
#         history_batch = history[i].numpy().reshape(-1, 1)
#         label_batch = label[i].numpy().reshape(-1, 1)
#         # 对每个 batch 的数据进行缩放
#         history_transformed[i] = scaler.fit_transform(history_batch).reshape(seq_len, 1)
#         label_transformed[i] = scaler.transform(label_batch).reshape(pred_len, 1)
#         scalers.append(scaler)  # 保存每个 batch 的 scaler 以备后用
#     # 将数据转换回原始类型
#     history = np.array(history_transformed)
#     label = np.array(label_transformed)

#     seqs = history.copy()
#     print("seqs.shape", seqs.shape)

#     device = "cpu" if sys.platform == 'darwin' else 'cuda:0'

#     # model = get_model('Timer-LOTSA', device)
#     # model = get_model('Chronos-tiny', device)
#     model = get_model('Timer-UTSD', device)
#     # model = get_model('Uni2ts-small', device)
#     # model = get_model('MOIRAI-small', device)
#     t = time_start()
#     preds = model.forcast(seqs, pred_len)
#     log_time_delta(t, 'Preprocess')
#     print("preds.shape", preds.shape)

#     # 画个图吧
#     # 把batch内的数据画在不同的子图上
#     plt.figure(figsize=(12, 6))
#     for i in range(batch_size):
#         plt.subplot(batch_size, 1, i + 1)
#         plt.plot(np.arange(seq_len + pred_len), np.concatenate([seqs[i], preds[i]]), label='pred')
#         plt.plot(np.arange(seq_len + pred_len), np.concatenate([history[i], label[i]]), label='truth')
#         plt.legend()
#         from Timer.utils.metrics import metric

#         mae, mse, rmse, mape, mspe = metric(preds[i], label[i])
#         plt.title(f"mae={mae:.4f}, mse={mse:.4f}, rmse={rmse:.4f}, mape={mape:.4f}, mspe={mspe:.4f}")
#     plt.show()
