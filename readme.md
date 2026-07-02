进程 1：run_loop.py
  - 拥有 LIBEROSimulationRuntime
  - 拥有 HTTPVLAClient
  - 负责推进 LIBERO 仿真

进程 2：dummy_vla_server.py / 真实 VLA server
  - 拥有 VLA 模型
  - 接收 observation
  - 返回 action