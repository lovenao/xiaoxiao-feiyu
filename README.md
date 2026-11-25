# 小小肥鱼 1.0

这是小小肥鱼 1.0 的历史源码快照，仅保留可公开的程序文件。

## 运行

Windows 下双击 `启动.bat`，或在本目录运行：

```powershell
python app.py
```

程序默认监听 `http://127.0.0.1:8765`。首次运行后，在设置页添加自己的供应商、模型和 API Key；配置会写入本地 `config.json`，该文件不会提交到 Git。

## 数据说明

此目录不包含原始配置、API Key、对话记录、缓存或长期记忆。`config.example.json` 仅展示配置结构。
