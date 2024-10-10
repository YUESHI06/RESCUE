noise_rates=(0.3)
vuls=("reentrancy" "timestamp")
device="$1"
# vuls=("timestamp")

for vul in "${vuls[@]}"
do
    for noise_rate in "${noise_rates[@]}"
    do
    # 对每个噪声值运行十次
    for i in {1..10}
    do
        python Fed_ARFL.py --vul $vul --noise_type fn_noise --noise_rate $noise_rate --device "$device"
    done
    done
done