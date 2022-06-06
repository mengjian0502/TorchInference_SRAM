PYTHON="/home/jmeng15/anaconda3/bin/python"

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
fi

export CUDA_VISIBLE_DEVICES=0

# setting
epochs=200
lr=0.1
batch_size=128
loss=cross_entropy
wbit=4
abit=4
dataset=cifar10
model=vgg7_quant

# dataset
data_path="./dataset/"
save_path="./save/${dataset}/${model}_w${wbit}_a${abit}/"
log_file="cnn_training.log"

$PYTHON -W ignore main.py \
    --model ${model} \
    --save_path ${save_path} \
    --epochs ${epochs} \
    --log_file ${log_file} \
    --lr  ${lr} \
    --loss_type ${loss} \
    --dataset ${dataset} \
    --batch_size ${batch_size} \
    --data_path ${data_path} \
    --wbit ${wbit} \
    --abit ${abit} \
    --ngpu 1;