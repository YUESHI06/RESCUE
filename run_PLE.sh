# noise_rates=(0.05 0.1 0.15 0.2 0.25 0.3)
noise_rates=(0.3)
vuls=("reentrancy" "timestamp")
device="$1"
# vuls=("timestamp")

for vul in "${vuls[@]}"
do
    for noise_rate in "${noise_rates[@]}"
    do
    # 对每个噪声值运行十次
    for i in {1..5}
    do
        python train.py --vul $vul --noise_type fn_noise --noise_rate $noise_rate --device cuda:3 --epoch 15 --outer_lr 0.003 --inner_lr 0.003 --device "$device"
    done
    done
done