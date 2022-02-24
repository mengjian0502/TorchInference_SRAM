# Pytorch-based SRAM inference simlator
@Jian Meng

## Basic CNN model

A 4-bit CNN with two layers are constructed as follow: 

```python
class CNN(nn.Module):
    def __init__(self, num_class=10, drop_rate=0.5, wbit=4, abit=4):
        super(CNN, self).__init__()
        self.conv1 = QConv2d(1, 128, 3, 1, bias=False, wbit=wbit, abit=abit)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = QConv2d(128, 128, 3, 1, bias=False, wbit=wbit, abit=abit)
        self.relu2 = nn.ReLU(inplace=True)

        self.dropout1 = nn.Dropout(drop_rate)
        self.dropout2 = nn.Dropout(drop_rate)
        self.fc1 = QLinear(4608, 128, wbit=wbit, abit=abit)
        self.fc2 = QLinear(128, num_class, wbit=wbit, abit=abit)
```

The quantization module `QConv2d` performs the 4-bit convolution for each layer. To run the training, execute the `run.sh` in your terminal:

```bash
bash run.sh
```

Before running, make sure your Python environment is correctly specified in the `run.sh`. Use `which python` to check the your current Python path. 

```bash
$ which python
"/home/userid/anaconda3/envs/myenv/bin/python"
```

The current 4-bit CNN can achieve 99.16% accuracy on MNIST dataset. The model checkpoint is available at: https://www.dropbox.com/sh/grtmeb4813ul144/AADv0-B2LHgbPijPD33hfBc2a?dl=0

## Running inference with the SRAM simulator

A SRAM simulator is implemented by replacing the `QConv2d` module with the `SRAMConv2d` :

```python
class SRAMCNN(nn.Module):
    def __init__(self, num_class=10, drop_rate=0.5, wbit=4, abit=4, subArray=64):
        super(SRAMCNN, self).__init__()
        self.conv1 = SRAMConv2d(1, 128, 3, 1, bias=False, wl_input=abit, wl_weight=wbit, subArray=subArray)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = SRAMConv2d(128, 128, 3, 1, bias=False, wl_input=abit, wl_weight=wbit, subArray=subArray)
        self.relu2 = nn.ReLU(inplace=True)

        self.dropout1 = nn.Dropout(drop_rate)
        self.dropout2 = nn.Dropout(drop_rate)
        self.fc1 = QLinear(4608, 128, wbit=wbit, abit=abit)
        self.fc2 = QLinear(128, num_class, wbit=wbit, abit=abit)
```

To start, specify the checkpoint directory and run the `inference.sh` in your terminal:

```bash
bash inference.sh
```

The pretrained weights will be mapped along the shape-wise dimension ([Peng, ISCAS, 2019](https://ieeexplore.ieee.org/document/8702715)), you can specify the size of the subarray size by changing the value of `subArray`  inside `inference.sh` (default = 64).

The low precision weights will be first decomposed into the binary, and the negative numbers are represented as the 2's complement format.

