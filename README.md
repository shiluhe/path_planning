## conda 环境
```bash
conda create -n pathplan python=3.12
conda activate pathplan
```

## 安装依赖
```bash
pip install -r requirements.txt
```

## 路径规划算法测试
```bash
./scripts/path_planning.sh \
    --init_method "$INIT_METHOD" \
    --opt_method "$OPT_METHOD" \
    --waypoints "$WAYPOINTS"
# init_method include:straight / dijkstra / a_star / rrt / genetic / aco / sa
# opt_method include: penalty / sqp
```

## 测试Example：
```bash
./scripts/path_planning.sh --init_method dijkstra --opt_method penalty --waypoints 150
```
