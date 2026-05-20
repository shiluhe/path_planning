## conda 环境
```bash
conda create -n pathplan python=3.12
conda activate pathplan
```

## 安装系统依赖（保存video）
```bash
sudo apt update
sudo apt install ffmpeg
```

## 安装环境依赖
```bash
pip install -r requirements.txt
```

## 二维路径规划算法测试
```bash
./scripts/path_planning.sh \
    --init_method "$INIT_METHOD" \
    --opt_method "$OPT_METHOD" \
    --waypoints "$WAYPOINTS"
# example: ./scripts/path_planning.sh --init_method dijkstra --opt_method penalty --waypoints 150
```
| 参数 | 含义 | 可选 |
|---|---|---|
| `--init_method` | 初始路径规划算法 | `straight / dijkstra / a_star / rrt / genetic / aco` |
| `--opt_method` | 数值优化算法 | `penalty / sqp` |
| `--output-dir` | 输出目录 | `results/ & video/` |

## 三维路径规划算法测试
```bash
python 3_Dimensional_Q/scripts/three_dimensional_path_planning.py \
    --segments 20 \
    --safety-margin 5
```
| 参数 | 含义 | 默认值 |
|---|---|---|
| `--segments` | 路径分段数 | `20` |
| `--safety-margin` | 障碍物外额外安全距离 | `5.0 m` |
| `--output-dir` | 输出目录 | `3_Dimensional_Q/results/three_dimensional` |



