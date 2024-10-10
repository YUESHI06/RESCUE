# noise_rates=(0.15 0.2 0.25 0.3)
noise_rates=(0.3)
device="$1"
vuls=("reentrancy" "timestamp")

for vul in "${vuls[@]}"
do
    for noise_rate in "${noise_rates[@]}"
    do
    # 对每个噪声值运行五次
    for i in {1..3}
    do
        python Ablation_PLE.py --vul $vul --noise_type fn_noise --noise_rate $noise_rate  --device cuda:4 --batch 8 --input_channel 428 --device "$device"
    done
    done
done