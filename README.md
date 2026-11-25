# 小小肥鱼 · 项目说明与维护文档

> 单文件、本地运行、面向个人使用的 OpenAI 兼容多轮问答 Agent。
>
> 本文档面向后续维护、排障和交接。涉及 `config.json`、`conversations.json`、`memory.json` 时，请先备份再修改。

---

## 仓库版本

- 当前程序版本：1.2.0
- 历史版本：通过 Git 提交历史查看 1.0 和 1.1

仓库根目录始终是当前版本，不再放置旧版目录。历史提交保留当时的程序源码和启动脚本；个人配置、API Key、对话记录、索引、缓存和长期记忆均不进入仓库。

---

## 1. 项目定位

小小肥鱼是一个本地 WebUI Agent，核心目标是：

- 只使用 Python 标准库运行，不引入 pip/npm 构建链
- 支持多个 OpenAI 兼容供应商
- 支持多轮对话、流式回答、工具调用和历史管理
- 尽量保持固定上下文前缀，提高供应商自动上下文缓存命中率
- 数据和 API Key 默认只保存在本机项目目录

它不是多人服务，也没有账号、权限和鉴权系统。默认只监听 `127.0.0.1`。

---

## 2. 快速开始

### 环境要求

- Windows 10/11
- Python 3.10+，当前开发环境为 Python 3.13
- 可访问模型供应商接口
- 浏览器需要联网加载 KaTeX CDN；数学公式和化学式渲染依赖它

### 启动

直接双击：

```text
启动.bat
```

脚本会依次寻找：

1. `py -3`
2. `python`

也可以手动运行：

```powershell
python app.py
```

可选参数：

```powershell
python app.py --port 8765
python app.py --port 8765 --no-browser
```

默认地址：

```text
http://127.0.0.1:8765
```

如果端口被占用，程序会自动尝试后续 29 个端口。

### 从 Git 仓库获取后首次运行

仓库只包含 `app.py`、`README.md` 和 `启动.bat`，不包含个人运行数据。首次运行时缺少 `config.json`、`conversations.json` 和 `memory.json` 不影响启动；程序会使用空供应商列表和内置提示词。

启动后进入设置页添加供应商、模型和 API Key。保存时只会在本机生成 `config.json`。配置、对话和记忆文件都只保留在本机，不应提交到仓库。

### 修改后刷新

- 修改后端 Python 代码：完全关闭启动窗口并重新运行 `启动.bat`
- 只修改内嵌 HTML/CSS/JS：重启后按 `Ctrl+F5`，避免浏览器缓存旧页面

---

## 3. 目录与文件职责

```text
小小肥鱼/
├─ app.py                 # 唯一程序文件：后端 + HTML/CSS/JS
├─ 启动.bat               # Windows 启动脚本
├─ README.md              # 本文档
├─ config.json            # 本机运行数据，不进入仓库
├─ conversations.json     # 本机对话数据，不进入仓库
├─ conversation_index.json # 本机历史索引，不进入仓库
├─ conversation_cache/    # 本机会话缓存，不进入仓库
└─ memory.json            # 本机记忆数据，不进入仓库
```

### 重要说明

- `app.py` 是单文件应用，后端、页面、CSS 和 JavaScript 全部内嵌。
- `config.json`、`conversations.json`、`memory.json` 都是运行时数据，不属于源码。
- `config.json` 中包含明文 API Key，禁止直接分享、提交到 Git 或上传云端。
- `conversations.json` 可能包含图片/PDF 页面图片的 Base64 数据，文件会快速变大。
- `conversation_index.json` 保存标题、元数据和可搜索正文；`conversation_cache/` 保存按会话拆分的完整副本。两者都包含私人数据。
- 历史列表和搜索读取轻量索引，打开单个对话读取对应缓存文件，避免每次都解析完整大文件。

---

## 4. 总体架构

```text
Browser
  │
  │ HTTP + JSON / SSE
  ▼
ThreadingHTTPServer
  │
  ├─ 配置与供应商管理
  ├─ 对话与消息持久化
  ├─ 流式模型请求
  ├─ 工具调用循环（联网搜索 / 记忆）
  ├─ 文件解析
  ├─ 上下文压缩
  └─ 记忆扫描与首轮自动标题
       │
       ├─ config.json
       ├─ conversations.json
       └─ memory.json
```

### 后端特点

- 使用 `ThreadingHTTPServer`，允许不同对话并行生成。
- 使用 SSE 将推理、正文、工具事件、错误和完成事件流式推送到浏览器。
- 所有 JSON 文件通过临时文件替换方式原子写入。
- 代码中没有数据库，状态完全由三个 JSON 文件承载。

### 前端特点

- 单页应用，无前端框架和构建工具。
- 主要状态集中在内嵌 JavaScript 的 `state` 对象。
- DOM 引用集中在 `el` 对象。
- 对话、设置、记忆、归档、弹窗均通过原生 DOM 渲染。
- 数学公式优先使用 KaTeX + mhchem；失败时使用内置降级渲染器。

---

## 5. 主要代码区域

### 后端

| 功能 | 主要函数/区域 |
|---|---|
| 配置读取与保存 | `load_config` / `save_config` / `default_settings` |
| 供应商与模型 | `provider_by_id` / `find_model` / `_handle_save_provider` |
| 标题截断与清理 | `make_title` / `clean_generated_title` |
| 流式模型调用 | `stream_chat` / `_open_chat_stream` |
| 工具循环 | `run_tool_loop` |
| 上下文构造 | `build_upstream_messages` |
| 上下文压缩 | `compress_context` 相关函数 |
| 文件解析 | `extract_file_text` / `_extract_pdf_content` / `_extract_docx_text` |
| 记忆 | `load_memory` / `_memory_scan_worker` / `build_memory_snapshot` |
| 首轮自动标题 | `generate_first_turn_title` |
| 对话请求主流程 | `HarnessHandler._handle_chat` |
| HTTP 路由 | `HarnessHandler.do_GET/do_POST/do_PATCH/do_DELETE` |

### 前端

| 功能 | 主要函数/区域 |
|---|---|
| 全局状态 | `state` |
| DOM 引用 | `el` |
| 对话列表 | `loadConversations` / `renderHistory` |
| 对话加载 | `openConversation` |
| SSE 读取 | `streamAssistant` / `readSSE` |
| 消息渲染 | `renderMessages` / `messageHtml` |
| Markdown/公式 | `renderMarkdown` / `renderMath` |
| 设置页 | `renderProviderGrid` / `renderMemorySettings` 等 |
| 图片查看器 | `viewImage` |
| 附件处理 | `addImageFiles` / `addFileAttachments` |

---

## 6. 一次对话的请求流程

1. 浏览器向 `/api/chat` 发送 JSON。
2. 后端校验供应商、模型、思考强度和提示词。
3. 新对话会：
   - 创建会话 ID
   - 绑定当前系统提示词原文
   - 生成日期时间环境信息
   - 若开启记忆，冻结一份记忆快照
   - 保存用户消息
4. 后端构造上游消息并返回 SSE。
5. 模型正文、推理和工具调用实时推送到前端。
6. 工具循环结束后保存助手消息。
7. 新对话第一轮成功完成时：
   - 使用“记忆供应商/记忆模型”生成短标题
   - 仅在成功且未被手动改名时替换临时标题
8. 发送 `done` 事件，前端刷新当前标题、历史列表和上下文指示。

### SSE 事件

常见事件类型：

- `meta`：会话 ID、标题、上下文信息
- `notice`：兼容模式提示
- `status`：当前步骤提示
- `reasoning`：推理内容
- `content`：正文增量
- `replace_content`：替换正文
- `tool_event`：工具调用状态
- `usage`：Token 使用量
- `error`：错误信息
- `done`：本轮完成

---

## 7. 上下文构造与缓存策略

`build_upstream_messages()` 的消息顺序是刻意固定的：

```text
1. 固定工具规则 system
2. 当前对话冻结的记忆快照 system（如有）
3. 当前日期和时间 system
4. 用户选择的系统提示词原文 system
5. 上下文压缩摘要 system（如有）
6. 按原始顺序追加的历史 user/assistant 消息
```

### 为什么这样排

- 工具规则固定在最前面，工具开关不改变前缀文本。
- 记忆快照在对话创建时冻结，后续轮次不变化。
- 日期时间在对话创建时记录，不在每轮插入。
- 系统提示词绑定到对话，不随设置页修改而变化。
- 历史消息只追加，不修改旧内容。

### 修改代码时的注意事项

以下行为会破坏缓存命中：

- 每轮请求重新生成系统前缀
- 把时间戳写入历史消息
- 修改旧消息文本
- 在固定前缀中间插入动态工具状态
- 每轮重新排列历史

如果必须改变前缀，应明确接受缓存失效。

---

## 8. 供应商、模型与思考强度

`config.json` 中供应商结构大致为：

```json
{
  "id": "p_xxx",
  "name": "供应商名称",
  "base_url": "https://example.com/v1",
  "api_key": "sk-...",
  "models": [
    {
      "id": "model-id",
      "name": "展示名称",
      "efforts": ["low", "medium", "high"],
      "default_effort": "medium",
      "context_window": 128000,
      "max_output_tokens": 8192
    }
  ],
  "timeout": 120,
  "retries": 3
}
```

### 兼容策略

`stream_chat()` 在供应商不支持某些参数时，会按顺序尝试兼容请求：

1. 标准请求
2. 去掉工具参数
3. 去掉 `stream_options`
4. 去掉 `reasoning_effort`

### 思考强度

- 若模型声明了 `efforts`，只允许使用声明档位。
- 若模型没有声明，前端仍允许显示通用档位。
- 显示标签统一使用“思考强度”。

---

## 9. 系统提示词与自动标题

### 系统提示词

- 新对话开始时选择一条提示词。
- 第一条消息发送后锁定。
- 对话保存 `prompt_id`、`prompt_name` 和当时的 `system_prompt` 原文。
- 修改设置页中的提示词预设，不影响已有对话。
- 对话页顶部会显示已锁定的提示词名称。
- 提示词内容允许完全为空；内置“空白”提示词不会注入任何角色 system 消息。

### 首轮自动标题

- 仅对新对话第一轮成功问答执行。
- 使用记忆设置中的供应商和模型。
- 标题长度由 `make_title` 限制，并清理 Markdown、引号和结尾标点。
- 失败时保留临时标题，不阻塞后续对话。
- 不会在第二轮、重新生成或编辑问题时重复执行。
- 若用户在生成期间已手动改名，不覆盖手动标题。

---

## 10. 记忆系统

记忆系统由四部分构成：

| 概念 | 存储位置 | 作用 |
|---|---|---|
| 用户画像 | `memory.json.profile` | 长期稳定偏好，最多 2500 字；默认按 Markdown 分类整理，超长不截断 |
| 对话索引 | `memory.json.conversations` | 全局、可搜索、可编辑的对话摘要；每条摘要最多 300 字 |
| 检查状态 | `memory.json.reviews` | 防止无价值对话重复判断，记录失败重试 |
| 对话快照 | `conversations.json[*].memory_snapshot` | 每个对话创建时冻结注入的系统记忆文本 |

画像默认使用固定分类：基本信息、偏好与习惯、学习与工作方式、当前长期目标、重要约束。设置页可以编辑“记忆整理提示词（画像与索引）”并恢复默认值；该提示词同时控制画像分类和索引摘要。

### 关键区别

- **索引不是每轮直接注入的。**
- **真正注入的是会话自己的快照。**
- 索引是生成快照的数据源之一，也供记忆搜索工具使用。
- 快照跟随具体会话，不是独立文件或独立列表。

### 快照生成

新对话开启记忆时，会构造一份快照：

- 用户画像
- 最近最多 40 条有价值对话索引
- 不再对整个索引区做固定字数截断，每条索引使用完整摘要
- 已归档索引会标记“已归档”
- 被永久排除的会话不会进入新快照

快照一旦写入会话，后续轮次保持不变。

### 索引

索引字段包括：

```json
{
  "id": "c_xxx",
  "title": "跟随对话标题",
  "summary": "不超过 300 字的摘要",
  "keywords": ["关键词"],
  "notes": [],
  "turns": 6,
  "archived": false,
  "updated_at": "...",
  "last_activity_at": "...",
  "last_memory_at": "...",
  "last_memory_message_count": 12
}
```

规则：

- 索引 ID 使用会话内部 ID，重命名不会断链。
- **索引标题始终跟随对话标题，不能在索引编辑器里单独修改。**
- 索引编辑器只修改摘要和关键词。
- 模型生成的摘要超过 300 字时，会调用记忆模型重新压缩；压缩失败则记为失败任务。
- 删除索引表示把该会话永久隔离出记忆系统。
- 隔离后：
  - 从当前索引移除
  - 不再参与记忆扫描
  - 不进入新对话快照
  - 不进入记忆搜索
  - 已有旧快照不修改

### 检查状态与失败重试

`reviews` 以会话 ID 为键，典型字段：

```json
{
  "reviewed_message_count": 12,
  "reviewed_at": "...",
  "worth_remembering": false,
  "failures": 0,
  "last_error": "",
  "last_failed_at": null,
  "pending_message_count": 0,
  "last_activity_at": "..."
}
```

规则：

- 无论模型判断“有价值”还是“无价值”，成功检查后都会推进已检查消息数。
- 只有出现新内容后才重新进入待检查状态。
- 新问答、重新生成回答、编辑问题都会增加消息并触发再次检查；单纯切换已存在的回答版本不会。
- 模型调用失败、返回非法 JSON 等不会推进已检查进度。
- 失败次数会累计，后续每次扫描继续重试。
- 成功后才清零失败状态。

### 扫描触发与条件

当前没有独立定时器线程。扫描触发点：

- 新建对话时
- 设置页点击“立即检查更新”时

候选会话必须满足：

- 未归档
- 未被排除
- 至少一轮消息
- 最后活动时间超过 30 分钟
- 当前消息数大于上次成功检查的消息数
- 记忆整理在约 48000 字预算内尽量选择更多最近轮次，不再固定只看最近 4 轮

单次最多处理 20 条。

### 记忆工具

模型可使用：

- `memory_search`：搜索索引
- `memory_read_conversation`：读取某对话最近正文
- `memory_save`：写入画像或当前对话笔记

被排除的对话禁止写入对话记忆。

---

## 11. 联网搜索

使用 Firecrawl v2：

```text
https://api.firecrawl.dev/v2/search
```

### 设置

- 全局开关
- Firecrawl API Key
- 每次搜索返回数：1–50，默认 10

### 对话内行为

- 输入框地球按钮控制本轮是否允许搜索。
- 只有模型判断需要搜索时才调用 `web_search`。
- 工具循环最多 3 轮。
- 单条结果 Markdown 截断约 12000 字。
- 搜索结果总上下文上限约 36000 字。
- 搜索关闭但模型尝试调用时，会返回 `web_search_disabled` 工具结果，由模型解释。

---

## 12. 文件和图片

### 支持类型

- PDF
- DOCX
- TXT / MD / CSV / JSON / LOG / XML
- HTML / HTM
- 其他可解码文本文件

旧版 `.doc` 不支持。

### PDF 策略

- 优先提取文字。
- 有文字也有页面图片时，两者都发送给支持视觉的模型。
- 扫描版 PDF 如果没有文字，会把页面渲染成图片。
- 最多处理前 100 页。
- 没有本地 OCR。

### 限制

文件数量、累计文字量和图片数量都不再设置固定上限。前端会根据当前模型的 `context_window` 估算上下文占用，只有预计超出上下文时才跳过附件。

| 项目 | 限制 |
|---|---|
| 单文件传输大小 | 25 MB（防止单次请求拖垮本地服务） |
| 单张图片 data URL | 6 MB（防止单张图片无限膨胀） |
| 请求体上限 | 24 MB |
| PDF 页数 | 100 页 |

上下文估算使用字符特征近似 Token 数，最终是否超限仍以供应商实际 tokenizer 为准。

图片查看器支持滚轮缩放、拖拽和双击复位。

---

## 13. 通用设置与对话管理

通用设置页提供对话字体大小加减控制（80%–140%，每档 10%），偏好保存在浏览器 localStorage。右侧问题导航在桌面端始终保留，仅在窄屏触屏设备隐藏。


支持：

- 新建、重命名、归档、恢复、永久删除
- 停止生成
- 重新生成最后一轮
- 编辑最后一轮问题
- 多版本回答切换
- 不同对话并行生成
- 右侧问题导航
- 标题、正文和附件文字搜索（包含归档）
- 上下文压缩

删除对话前必须先归档。

---

## 14. API 一览

### GET

| 路径 | 说明 |
|---|---|
| `/api/health` | 健康检查 |
| `/api/providers` | 供应商列表（不返回完整 Key） |
| `/api/prompts` | 系统提示词 |
| `/api/memory` | 记忆设置、画像、索引、重试数 |
| `/api/memory/snapshots/{id}` | 读取单会话快照（内部兼容接口，UI 未使用） |
| `/api/settings` | 搜索设置 |
| `/api/conversations` | 对话列表（读取轻量索引） |
| `/api/conversations/search` | 按标题、正文和附件文字搜索历史对话 |
| `/api/conversations/{id}` | 单个对话完整内容 |

### POST

| 路径 | 说明 |
|---|---|
| `/api/providers` | 新建/更新供应商 |
| `/api/providers/{id}/refresh` | 刷新模型列表 |
| `/api/prompts` | 新建/更新提示词 |
| `/api/prompts/{id}` | 删除提示词 |
| `/api/memory` | 保存记忆系统配置 |
| `/api/memory/profile` | 保存用户画像 |
| `/api/memory/index/{id}` | 编辑索引摘要和关键词 |
| `/api/memory/scan` | 触发记忆扫描 |
| `/api/memory/snapshots/{id}` | 内部兼容保存快照，UI 未使用 |
| `/api/settings/web` | 保存联网搜索设置 |
| `/api/chat` | 发起对话（SSE） |
| `/api/conversations/{id}/compress` | 压缩上下文 |
| `/api/files/extract` | 上传并解析文件 |

### PATCH

| 路径 | 说明 |
|---|---|
| `/api/conversations/{id}` | 重命名、归档、切换回答版本 |

### DELETE

| 路径 | 说明 |
|---|---|
| `/api/providers/{id}` | 删除供应商 |
| `/api/conversations/{id}` | 永久删除已归档对话 |
| `/api/memory/index/{id}` | 删除索引并排除会话 |
| `/api/memory/snapshots/{id}` | 内部兼容删除快照，UI 未使用 |

---

## 15. 数据安全与备份

### 敏感信息

- `config.json` 中有明文 API Key。
- `conversations.json`、`memory.json` 中有聊天内容和长期记忆。
- 项目没有加密、登录和访问控制。

### 安全边界

- 服务绑定 `127.0.0.1`，默认不允许局域网访问。
- 不要把端口转发或反向代理到公网。
- 不要在截图、日志或测试脚本中输出完整 Key。

### 备份

建议在以下操作前手动复制整个项目目录：

- 大规模修改记忆结构
- 批量删除索引
- 迁移数据文件
- 大幅修改 `app.py`

JSON 写入使用临时文件替换，但这不是备份替代品。

---

## 16. 维护与扩展指南

### 新增 API

1. 在 `do_GET/do_POST/do_PATCH/do_DELETE` 中添加路由。
2. 编写 `_handle_*` 方法。
3. 所有配置/对话/记忆写入必须使用 `FILE_LOCK`。
4. 使用 `_atomic_write_json` 保存。
5. 前端通过 `api()` 调用。

### 修改记忆结构

- 先确认旧数据迁移策略。
- `load_memory()` 是规范化入口，新增字段应在这里兼容旧值。
- `memory.json.version` 当前仍为 `1`；若做不兼容迁移，应升级版本并写迁移代码。
- 迁移前备份 `memory.json`。

### 修改快照逻辑

- 快照在 `build_memory_snapshot()` 中生成。
- 快照只在会话创建时写入。
- 不要每轮重建快照，否则会破坏缓存。
- 被排除会话通过 `memory_excluded` 过滤。

### 修改上下文顺序

`build_upstream_messages()` 的顺序属于稳定协议。修改前必须考虑：

- 已有会话的缓存前缀
- 旧对话是否还能继续
- 上下文压缩摘要的位置
- 工具规则是否需要保持固定

### 测试注意事项

项目中不再附带测试目录。手工测试必须使用临时数据目录，尤其禁止让测试脚本写正式：

- `config.json`
- `conversations.json`
- `memory.json`

历史事故：测试配置曾覆盖正式 `config.json`，导致供应商列表丢失。以后任何会调用 `save_config()` 的测试，都必须先重定向 `CONFIG_PATH` 到临时目录。

---

## 17. 常见问题

### 页面显示“服务未连接”

- 检查 `启动.bat` 黑窗口是否仍在运行
- 查看端口是否被占用
- 打开启动窗口显示的 URL

### 供应商返回网页而不是模型

检查 Base URL：

- 有的中转站官网域名和 API 域名不同
- 通常需要 `https://.../v1`

### 公式显示红字

- 检查浏览器是否能访问 KaTeX CDN
- 刷新页面
- 复杂化学式依赖 mhchem

### 上下文超限

- 点击输入框上方/右下角的上下文按钮压缩
- 或新建对话继续

### 记忆没有更新

- 最后活动时间是否超过 30 分钟
- 是否有新消息
- 记忆供应商和模型是否仍存在
- 设置页是否显示失败待重试
- 会话是否已归档或被排除

### 索引删除后对话还在

这是预期行为。删除索引只隔离记忆系统，不删除聊天记录。

---

## 18. 已知限制

- 单文件已较大，继续增长会降低可维护性。
- 没有数据库，所有数据依赖 JSON 文件。
- 没有鉴权和加密。
- 没有本地 OCR。
- 记忆扫描不是独立定时任务，只在新建对话或手动检查时触发。
- KaTeX 依赖 CDN。
- 图片和页面图片用 Base64 存入对话，长期使用会让 `conversations.json` 很大。
- 旧版 `.doc` 不支持。

---

## 19. 快速验收清单

每次大改后至少检查：

- 启动窗口无 Python 异常
- 供应商数量、模型列表正常
- 新对话能流式回答
- 停止、重新生成、编辑问题正常
- 刷新后历史记录仍在
- 系统提示词锁定和显示正常
- 公式、化学式正常
- 图片上传、粘贴、滚轮缩放正常
- PDF/DOCX 提取正常
- 联网搜索开关正常
- 记忆画像可保存
- 索引标题跟随对话标题
- 索引摘要/关键词可编辑
- 删除索引后不会进入新快照
- 失败记忆任务会累计并重试
