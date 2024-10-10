noise_rates=(0.3)
# noise_rates=(0.05 0.1 0.15 0.2 0.25)
# noise_rates=(0.5 0.6 0.7 0.8 0.9 1.0)
vuls=("reentrancy" "timestamp")
# vuls=("reentrancy")
fracs=(0.2 0.4 0.6 0.8)
device="$1"


for vul in "${vuls[@]}"
do
    for noise_rate in "${noise_rates[@]}"
    do
    # 对每个噪声值运行五次
        for frac in "${fracs[@]}"
        do
            for i in {1..3}
            do
                python Fed_PLE.py --vul $vul --noise_type fn_noise --noise_rate $noise_rate  --device "$device" --batch 8 --input_channel 428 --valid_frac $frac --epoch 40  --warm_up_epoch 5
            done
        done
    done
done