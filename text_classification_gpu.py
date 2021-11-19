import re
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

class Data():
    def __init__(self, dataset_path, dev_ratio=0.2):
        with open(dataset_path, 'r', encoding='utf-8') as f:  # 读取文件
            lines = f.readlines()
            lines = [line.strip('\n').split(' ', 2) for line in lines]

        x_raw = [line[2] for line in lines]
        # x_raw=x_raw[:100]
        x_raw = np.array(x_raw)
        y = [int(line[0]) for line in lines]

        self.data_size = len(y)
        x_train, x_dev, y_train, y_dev = self._train_dev_split(x_raw, y, dev_ratio)  # 将数据分为训练集和验证集

        train_inputs, train_masks = self._preprocessing_for_bert(x_train)
        dev_inputs, dev_masks = self._preprocessing_for_bert(x_dev)
        self.train_data = TensorDataset(train_inputs, train_masks, torch.tensor(y_train))
        self.dev_data = TensorDataset(dev_inputs, dev_masks, torch.tensor(y_dev))

    def _train_dev_split(self, x, y, dev_ratio):
        train_size = int(self.data_size * (1 - dev_ratio))
        x_train = x[:train_size]
        y_train = y[:train_size]
        x_dev = x[train_size:]
        y_dev = y[train_size:]
        return x_train, x_dev, y_train, y_dev

    def _text_preprocessing(self, text):
        text = re.sub(r'(@.*?)[\s]', ' ', text)  # Remove '@name'
        text = re.sub(r'&amp;', '&', text)  # Replace '&amp;' with '&'
        text = re.sub(r'\s+', ' ', text).strip()  # Remove trailing whitespace
        return text

    def _preprocessing_for_bert(self, data):
        input_ids = []
        attention_masks = []

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        for sent in data:
            encoded_sent = tokenizer.encode_plus(
                text=self._text_preprocessing(sent),  # 句子文本预处理
                add_special_tokens=True,  # 添加 `[CLS]`和 `[SEP]`标记
                max_length=64,  # 截断/填充的最大长度
                pad_to_max_length=True,  # Pad sentence to max length
                # return_tensors='pt',           # Return PyTorch tensor
                return_attention_mask=True,  # Return attention mask
                truncation=True
            )

            # 将结果添加到list中
            input_ids.append(encoded_sent.get('input_ids'))
            attention_masks.append(encoded_sent.get('attention_mask'))

        # 将list转换为tensors
        input_ids = torch.tensor(input_ids)
        attention_masks = torch.tensor(attention_masks)

        return input_ids, attention_masks

    def get_train_dataset(self, batch_size=100, shuffle=True):
        return DataLoader(self.train_data, batch_size=batch_size, shuffle=shuffle)

    def get_dev_dataset(self, batch_size=100, shuffle=True):
        return DataLoader(self.dev_data, batch_size=batch_size, shuffle=shuffle)




def train(model, train_dataset, dev_dataset, lr=0.01, batch_size=64, epoch_num=10, use_GPU=False):
    if use_GPU:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    train_loss = []
    train_acc = []
    dev_loss = []
    dev_acc = []

    print("start training.....")
    for epoch in range(epoch_num):
        epoch_start_time = time.time()
        epoch_train_loss = 0
        epoch_dev_loss = 0
        epoch_train_acc = 0
        epoch_dev_acc = 0

        model.train()
        with tqdm(
                iterable=train_dataset,
                bar_format='{desc} {n_fmt:>4s}/{total_fmt:<4s} {percentage:3.0f}%|{bar}| {postfix}',
        ) as t:
            for i, data in enumerate(train_dataset):
                t.set_description_str(f"\33[36m【Epoch {epoch + 1:04d}】")
                optimizer.zero_grad()
                batch_ids, batch_masks, batch_label = tuple(t for t in data)
                y_pred = model(batch_ids.to(device), batch_masks.to(device))
                batch_loss = loss_fn(y_pred, batch_label.to(device))
                batch_loss.backward()
                optimizer.step()
                epoch_train_loss += batch_loss.item()
                label_pred = np.argmax(y_pred.cpu().data.numpy(), axis=1)
                acc = np.sum(label_pred == batch_label.numpy())
                epoch_train_acc += acc
                cur_time = time.time()
                delta_time = cur_time - epoch_start_time
                t.set_postfix_str(
                    f"train_loss={epoch_train_loss / train_dataset.__len__() / batch_size:.6f}， 执行时长：{delta_time}\33[0m")
                t.update()
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(dev_dataset):
                batch_ids, batch_masks, batch_label = tuple(t for t in data)
                y_pred = model(batch_ids.to(device), batch_masks.to(device))
                batch_loss = loss_fn(y_pred, batch_label.to(device))
                epoch_dev_loss += batch_loss.item()
                label_pred = np.argmax(y_pred.cpu().data.numpy(), axis=1)
                acc = np.sum(label_pred == batch_label.numpy())
                epoch_dev_acc += acc
        train_loss.append(epoch_train_loss / train_dataset.__len__() / batch_size)
        dev_loss.append(epoch_dev_loss / dev_dataset.__len__() / batch_size)
        train_acc.append(epoch_train_acc / train_dataset.__len__() / batch_size)
        dev_acc.append(epoch_dev_acc / dev_dataset.__len__() / batch_size)
        print(
            "time: {:.2f}, epoch {:.2f} train loss: {:.2f},dev loss: {:.2f}, train acc: {:.2f}, dev acc: {:.2f}".format(
                time.time() - epoch_start_time, epoch, train_loss[-1], dev_loss[-1], train_acc[-1], dev_acc[-1]))
    torch.save(model.state_dict(), '11_16.pth')
    return train_loss, dev_loss, train_acc, dev_acc
class BertClassifier(nn.Module):
    def __init__(self, freeze_bert=True):
        super(BertClassifier, self).__init__()
        D_in, H, D_out = 768, 50, 2

        self.bert = BertModel.from_pretrained('bert-base-uncased')  # 实例化BERT模型

        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(H, D_out)
        )
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state_cls = outputs[0][:, 0, :]  # 为分类任务提取标记' [CLS] '的最后一个隐藏状态
        logits = self.classifier(last_hidden_state_cls)
        return logits


data_file_path = 'data/training_label.txt'
batch_size = 1024
epoch_num = 2
lr = 0.01
use_GPU = True

data = Data(data_file_path)
train_dataset = data.get_train_dataset(batch_size=batch_size)
dev_dataset = data.get_dev_dataset(batch_size=batch_size)
model = BertClassifier()

train_acc, dev_acc, train_loss, dev_loss = train(model, train_dataset, dev_dataset, lr=lr, batch_size=batch_size,
                                                 epoch_num=epoch_num, use_GPU=use_GPU)
t = np.arange(1, len(train_loss) + 1)
acc_plot = plt.subplot(2, 2, 1)
plt.title('acc')
plt.plot(t, train_acc, color='red', label='train acc')
plt.plot(t, dev_acc, color='blue', label='dev acc')
loss_plot = plt.subplot(2, 2, 2)
plt.title('loss ')
plt.plot(t, train_loss, color='red', label='train loss')
plt.plot(t, dev_loss, color='skyblue', label='dev loss')
plt.show()
