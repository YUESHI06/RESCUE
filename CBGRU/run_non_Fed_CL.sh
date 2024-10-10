noise_rates=(0.3)
# vuls=("reentrancy" "timestamp")
vuls=("reentrancy" "timestamp")
device="$1"


for vul in "${vuls[@]}"
do
    for noise_rate in "${noise_rates[@]}"
    do
    # 对每个噪声值运行五次
    for i in {1..10}
    do
        python non_Fed_CL.py --vul $vul --noise_type diff_noise --noise_rate $noise_rate  --device "$device" --batch 8 --input_channel 428
    done
    done
done