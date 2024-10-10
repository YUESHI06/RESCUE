device="$1"
noise_rates=(0.3)
vuls=("reentrancy" "timestamp")
# vuls=("timestamp")

for vul in "${vuls[@]}"
do
    for noise_rate in "${noise_rates[@]}"
    do
    # 对每个噪声值运行十次
    for i in {1..3}
    do
        python Fed_Corr.py --vul $vul --noise_type fn_noise --noise_rate $noise_rate --device "$device"
    done
    done
done