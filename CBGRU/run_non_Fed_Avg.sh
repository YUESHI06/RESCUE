# noise_rates=(0.05 0.1 0.15 0.2 0.25 0.3)
noise_rates=(0.3)
vuls=("reentrancy" "timestamp")
device="$1"

for vul in "${vuls[@]}"
do
    for noise_rate in "${noise_rates[@]}"
    do
    for i in {1..10}
    do
        python non_Fed_Avg.py --vul $vul --noise_type diff_noise --noise_rate $noise_rate --cbgru_net2 bigru --device "$device"
    done
    done
done