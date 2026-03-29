## GoT Realtime Visualizer (Vue)

这是一个独立的 Vue3 + Vite 前端，用于连接你后端的实时可视化服务（默认 `http://127.0.0.1:8765`），并动态渲染推理图。

### 功能

- 方法选择：IO / COT / TOT / GOT / multiAgentGoT
- 输入题目 ID（默认自动填充最新 run）
- 实时轮询 `/events` 增量渲染
- 点击节点查看：
  - 结论（默认展开）
  - 原始输出/思考内容（可折叠）
- 思考动画：
  - **旋转**：节点外圈旋转
  - **呼吸灯**：节点发光/呼吸

### 运行方式

1) 启动后端实时服务（你现有的 python 入口带 `--realtime_vis` 即可），例如：

```bash
python multi_hop_qa/multi_hop_qa.py --dataset musique_ans --num_samples 1 --realtime_vis --vis_host 127.0.0.1 --vis_port 8765
```

2) 启动前端

```bash
cd frontend
npm install
npm run dev
```

3) 打开浏览器输出的地址（Vite 默认 `http://127.0.0.1:5173`）。

### 说明

- 默认通过 Vite dev server 的代理访问后端：`/api/* -> http://127.0.0.1:8765/*`
- 若你改了后端端口，修改 `vite.config.js` 里的 proxy target 即可。
