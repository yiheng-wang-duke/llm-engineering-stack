# 验证记录

验证环境：Python 3.11.6；独立虚拟环境 `.venv`；运行依赖版本锁定在 `requirements.lock`。

## 已执行

最终结果：**30 项自动化测试通过**；Ruff 检查通过；pip check 无依赖冲突；真实 HTTP 演示与 Chromium 冒烟通过。Compose YAML 已静态解析，未运行容器。

- 应用与协议自动化测试：上传、去重、文档路径标签处理、检索、跨会话记忆、重启持久化、导出、级联删除。
- 安全与边界测试：访问令牌、跨来源写入、超大文件、损坏 PDF、纯扫描 PDF、空白输入、计算器表达式限制、工具开关。
- 模型协议测试：真实 VLLM HTTP 适配器连接 mock transport，覆盖分段工具参数、工具结果回传、token usage、流中断、长度截断、工具错误、重复调用及最终轮预算。
- 向量协议测试：query instruction、响应顺序、归一化、缺失向量、维度变化、索引配置变化、重建失败回滚。
- 真实网络取消测试：启动 Uvicorn，客户端断开正在生成的 SSE，确认上游任务取消、会话释放、run 标为 cancelled、未写入不完整历史。
- 真实 HTTP 演示冒烟：启动独立服务，完成上传 → 检索 → 记忆 → SSE → 历史持久化，并清理临时记录。
- Chromium：导入文档、检索、预览、保存记忆、对话、来源、导出 JSON、刷新后历史回看；390px 手机布局；没有 JavaScript 未捕获错误。
- Ruff 静态检查与格式化。

## 截图

- [桌面](screenshots/desktop.png)
- [对话与来源](screenshots/conversation.png)
- [手机](screenshots/mobile.png)

## 尚未验证

当前执行节点没有可用 NVIDIA GPU，也没有 Docker 命令。因此未下载 Qwen 权重、未启动真实 vLLM GPU 引擎、未测量显存/性能/模型回答质量，未构建或启动容器。

mock 协议测试和 demo 端到端测试不等于真实模型验证。部署到 GPU 机器后运行：

```bash
.venv/bin/python scripts/smoke.py --require-vllm
```

并在 UI 要求模型调用计算工具与知识库检索，检查执行记录；开启 Embedding 后重新索引并验证中文同义查询的检索质量。

Starlette 测试客户端当前会报告关于未来 HTTP 客户端切换的弃用提示，测试仍通过。
