PYTHON="/home/jmeng15/anaconda3/envs/myenv/bin/python"

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
fi

export CUDA_VISIBLE_DEVICES=1

# setting
epochs=100
lr=0.1
batch_size=128
loss=cross_entropy
wbit=4
abit=4

# dataset
data_path="./dataset/"
save_path="./save/mnist_train/cnn_w${wbit}_a${abit}/"
log_file="cnn_training.log"

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
    --ngpu 1;