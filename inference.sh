PYTHON="/home/jmeng15/anaconda3/bin/python"

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
fi

export CUDA_VISIBLE_DEVICES=0

# setting
epochs=30
lr=0.1
batch_size=128
loss=cross_entropy
subArray=64
rram_type=6K
wbit=4
abit=4
dataset=cifar10
model=sram_vgg7


# dataset
data_path="./dataset/"
save_path="./save/cifar10/vgg7_quant_w4_a4/eval/"
log_file="cnn_eval.log"

# checkpoint
# pretrained_model="./save/cifar10/vgg7_quant_w4_a4/model_best.pth.tar"
pretrained_model="/home/jmeng15/ICCAD_2022_PRIVE/TorchInference_RRAM/save/vgg7_quant/vanilla/vgg7_quant_w4_a4_mode_sawb_wd1e-4/model_best.pth.tar"

$PYTHON -W ignore main.py \
    --model ${model} \
    --save_path ${save_path} \
    --dataset ${dataset} \
    --epochs ${epochs} \
    --log_file ${log_file} \
    --lr  ${lr} \
    --rram_type ${rram_type} \
    --loss_type ${loss} \
    --sensitive_lv 0.0 \
    --batch_size ${batch_size} \
    --data_path ${data_path} \
    --wbit ${wbit} \
    --abit ${abit} \
    --subArray ${subArray} \
    --fine_tune \
    --resume ${pretrained_model} \
    --evaluate \
    --ngpu 1;