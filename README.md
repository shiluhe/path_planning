## conda 环境
conda create -n pathplan python=3.12
conda activate pathplan

## 安装依赖
pip install -r requirements.txt

## 路径规划算法测试
./scripts/path_planning.sh \
    --init_method "$INIT_METHOD" \
    --opt_method "$OPT_METHOD" \
    --waypoints "$WAYPOINTS"

## Example：
./scripts/path_planning.sh --init_method dijkstra --opt_method penalty --waypoints 150