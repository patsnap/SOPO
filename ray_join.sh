export MASTER_IP=10.148.0.108
export MASTER_PORT=23456
export GPUS_PER_NODE=8
ray stop
ray start --address=$MASTER_IP:$MASTER_PORT --node-ip-address=$(hostname -I | awk '{print $1}') --num-gpus=$GPUS_PER_NODE