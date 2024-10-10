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
    for i in {1..3}
    do
        # python train.py --vul $vul --noise_type fn_noise --noise_rate $noise_rate --device "$device" --outer_lr 0.002 --inner_lr 0.002 --epoch 20 
        python warm_up_PLE.py --vul $vul --noise_type fn_noise --noise_rate $noise_rate --device "$device" --epoch 20 
    done
    done
done