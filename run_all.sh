#!/bin/bash

# ====================== 请根据你的环境修改以下参数 ======================
PYTHON_SCRIPT="main.py"  # 你的Python代码文件名（如 main.py）
CONDA_ENV="timerxl"         # 你的Python环境名（如conda/venv环境）
LOG_DIR="./search_logs"         # 日志保存目录（自动创建）
# ======================================================================

# 1. 创建日志目录（若不存在）
mkdir -p $LOG_DIR

# 2. 定义8个GPU与8个数据集的对应关系（顺序可自行调整）
# 格式："GPU编号 数据集名称"
declare -a tasks=(
    "0 ETTh1"
    "1 ETTh2"
    "2 ETTm1"
    "3 ETTm2"
    "4 Exchange"
    "5 Weather"
    "6 Electricity"
    "7 Traffic"
)

# 3. 批量启动每个任务（每个任务一个tmux会话）
for task in "${tasks[@]}"; do
    # 拆分GPU编号和数据集
    GPU=$(echo $task | cut -d' ' -f1)
    DATA=$(echo $task | cut -d' ' -f2)
    # tmux会话名（格式：ts_gpu{GPU}_{DATA}，如 ts_gpu0_ETTh1）
    SESSION_NAME="ts_gpu${GPU}_${DATA}"
    # 日志文件名（每个任务独立日志）
    LOG_FILE="${LOG_DIR}/${SESSION_NAME}.log"

    echo "=== 启动任务：GPU ${GPU} + 数据集 ${DATA}，会话名：${SESSION_NAME} ==="

    # 创建tmux会话并在其中运行任务
    tmux new-session -d -s $SESSION_NAME <<EOF
# 激活Python环境（若用venv，替换为 source /path/to/venv/bin/activate）
conda activate $CONDA_ENV
# 进入代码所在目录（若脚本不在代码目录，需先cd，如 cd /path/to/your/code）
# cd /home/yourname/time_series_project
# 运行Python代码，日志写入LOG_FILE
python $PYTHON_SCRIPT --gpu $GPU --data $DATA > $LOG_FILE 2>&1
# 任务结束后不自动关闭会话（便于后续查看日志）
exec bash
EOF

    # 等待1秒，避免同时启动8个任务导致GPU瞬间负载过高
    sleep 1
done

echo -e "\n=== 所有任务已启动！共8个GPU，8个数据集 ==="
echo "查看所有tmux会话：tmux ls"
echo "进入某任务会话：tmux attach -t ts_gpu{编号}_{数据集}（如 tmux attach -t ts_gpu0_ETTh1）"
echo "查看某任务日志：cat ${LOG_DIR}/ts_gpu{编号}_{数据集}.log（如 cat ${LOG_DIR}/ts_gpu0_ETTh1.log）"