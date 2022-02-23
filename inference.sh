PYTHON="/home/jmeng15/anaconda3/envs/myenv/bin/python"

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
fi

export CUDA_VISIBLE_DEVICES=1

# setting
epochs=30
lr=0.1
batch_size=128
loss=cross_entropy
subArray=64
wbit=4
abit=4

# dataset
data_path="./dataset/"
save_path="./save/mnist_eval/cnn_w${wbit}_a${abit}_subArray${subArray}/"
log_file="cnn_eval.log"

# checkpoint
pretrained_model="./save/mnist_train/cnn_w4_a4/model_best.pth.tar"

$PYTHON -W ignore main.py \
    --save_path ${save_path} \
    --epochs ${epochs} \
    --log_file ${log_file} \
    --lr  ${lr} \
    --loss_type ${loss} \
    --batch_size ${batch_size} \
    --data_path ${data_path} \
    --wbit ${wbit} \
    --abit ${abit} \
    --subArray ${subArray} \
    --fine_tune \
    --resume ${pretrained_model} \
    --evaluate \
    --ngpu 1;