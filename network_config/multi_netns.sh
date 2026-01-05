#!/bin/bash

# 网络命名空间配置脚本 - 为每个终端创建隔离网络环境

# 清理函数
cleanup() {
    echo "清理网络命名空间..."
    for ns in net1 net2 net3; do
        sudo ip netns delete $ns 2>/dev/null
    done
    # 删除网桥会自动删除连接在上面的 veth 接口
    sudo ip link set br-test down 2>/dev/null
    sudo ip link delete br-test type bridge 2>/dev/null
    exit
}
trap cleanup SIGINT

# 初始化网桥
setup_bridge() {
    echo "创建虚拟网桥 br-test..."
    sudo ip link add name br-test type bridge
    sudo ip addr add 192.168.100.1/24 dev br-test
    sudo ip link set br-test up
    # 允许网桥转发流量（某些系统需要）
    sudo iptables -A FORWARD -i br-test -j ACCEPT 2>/dev/null
    sudo iptables -A FORWARD -o br-test -j ACCEPT 2>/dev/null
}

# 创建网络命名空间
create_netns() {
    local ns=$1
    local veth_host="veth${ns##net}-host"
    local veth_ns="veth${ns##net}-ns"
    local ip_addr="192.168.100.${ns##net}0/24"
    local delay=$2
    local bandwidth=$3

    echo "创建网络命名空间: $ns"

    # 创建命名空间
    sudo ip netns add $ns

    # 创建 veth pair
    sudo ip link add $veth_host type veth peer name $veth_ns

    # 将一端移到命名空间
    sudo ip link set $veth_ns netns $ns

    # 将主机端连接到网桥
    sudo ip link set $veth_host master br-test
    sudo ip link set $veth_host up

    # 配置命名空间端
    sudo ip netns exec $ns ip addr add $ip_addr dev $veth_ns
    sudo ip netns exec $ns ip link set $veth_ns up
    sudo ip netns exec $ns ip link set lo up

    # 添加路由
    sudo ip netns exec $ns ip route add default via 192.168.100.1

    # 配置延迟和带宽 (在主机端接口上配置，控制发往命名空间的流量)
    # 注意：在网桥模式下，tc 依然可以应用在成员接口上
    sudo tc qdisc add dev $veth_host root netem delay ${delay}ms

    echo "网络命名空间 $ns 配置完成 (IP: ${ip_addr%/*}, 延迟: ${delay}ms)"
}

# 初始化
setup_bridge

# 创建三个命名空间，每个有不同配置
create_netns net1 10 1024   # 1ms delay, 1Gbps
create_netns net2 10 1024    # 1ms delay, 1Gbps
create_netns net3 10 1024   # 1ms delay, 1Gbps

echo ""
echo "网络配置完成!"
echo "使用方法:"
echo "  终端1: sudo ip netns exec net1 bash"
echo "  终端2: sudo ip netns exec net2 bash"
echo "  终端3: sudo ip netns exec net3 bash"
echo ""
echo "在每个命名空间中测试网络:"
echo "  ping 192.168.100.1"
echo "  iperf3 -c 192.168.100.1 (如果安装了iperf3)"
echo ""
echo "按 Ctrl+C 清理"

# 保持运行
while true; do
    sleep 1
done