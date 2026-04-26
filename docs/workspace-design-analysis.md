# tyagent 工作空间设计分析

> 研究对象：hermes-agent、openclaw（hermes 前身）、tyagent（当前）
> 研究目的：为 tyagent 设计一个合理的"集中存放空间"体系

---

## 一、三家设计的全景对比

### 1.1 OpenClaw（hermes 前身）

OpenClaw 的核心概念是**"Home + Workspace"双层结构**：

```
~/.openclaw/                    # Home（全局配置、凭证）
├── config.json                 # 全局配置（模型、平台 token）
└── <workspace-name>/           # Workspace（每个工作区独立）
    ├── todo.json               # 当前任务列表
    ├── sessions/               # 会话历史
    ├── memory/                 # 记忆文件（按会话或全局）
    └── logs/                   # 运行日志
```

**关键洞察**：OpenClaw 的 Workspace 是按"项目"或"工作上下文"划分的，不是按用户。你在做项目 A 时的工作区记忆和任务，与做项目 B 时完全隔离。

### 1.2 Hermes（当前 hermes-agent）

Hermes 在 OpenClaw 的基础上做了重大重构，核心变化：

**（1）统一 Home 目录**：`~/.hermes`（可自定义 `HERMES_HOME`）

```
~/.hermes/                      # Home（单一入口）
├── config.yaml                 # 主配置
├── .env                        # 敏感凭证（不进入 git）
├── skills/                     # 技能（可复用的工作流）
│   └── <skill-name>/
│       ├── SKILL.md            # 技能定义
│       └── .../
├── memories/                   # 持久记忆
│   ├── MEMORY.md               # Agent 学到的环境事实
│   └── USER.md                 # 用户画像与偏好
├── cache/                      # 缓存
│   ├── images/                 # 下载的图片
│   └── .../
├── sessions/                   # 会话数据
├── home/                       # 子进程 HOME（隔离 git/ssh 配置）
└── .skills_prompt_snapshot.json # 技能索引快照
```

**（2）Profile 模式**：`~/.hermes/profiles/<name>/`

每个 Profile 有自己完整的 Home 目录，适合多用户/多角色场景：
- `profiles/coder/`：编程助手配置
- `profiles/writer/`：写作助手配置

**（3）工作区上下文文件**：`.hermes.md` / `HERMES.md`

在项目目录中放置 `.hermes.md`，hermes 会：
1. 从当前目录开始向上遍历
2. 直到找到 git root 或文件系统根
3. 将内容注入系统提示词

这让 agent 能感知"我现在在这个项目里工作"。

**（4）记忆系统**：`MEMORY.md` + `USER.md`

| 文件 | 用途 | 内容示例 |
|------|------|---------|
| `MEMORY.md` | Agent 的个人笔记 | "GitHub CLI 非交互式 auth 需要 GH_TOKEN 环境变量" |
| `USER.md` | 用户画像 | "用户偏好简洁回答，不喜欢冗长解释" |

约束：
- `MEMORY.md` 上限 2200 字符
- `USER.md` 上限 1375 字符
- 使用 `§` 作为条目分隔符
- 每次写入前扫描注入/渗透攻击模式
- 原子写入（temp + rename）+ 文件锁

**（5）系统提示词组装**：`prompt_builder.py`

系统提示词不是单一字符串，而是多个块的拼接：
1. `DEFAULT_AGENT_IDENTITY` — 基础身份
2. `MEMORY_GUIDANCE` — 记忆使用指导
3. `SESSION_SEARCH_GUIDANCE` — 会话搜索指导
4. `SKILLS_GUIDANCE` — 技能使用指导
5. `PLATFORM_HINTS[<platform>]` — 平台特定提示（Telegram 不支持 markdown 等）
6. `build_environment_hints()` — 环境检测（WSL 等）
7. `MEMORY.md` / `USER.md` 快照 — 持久记忆
8. `.hermes.md` 内容 — 工作区上下文
9. 技能索引 — 可用技能列表

### 1.3 tyagent（当前）

```
~/.tyagent/                    # Home（单一入口）
├── config.yaml                 # 配置（含凭证，0o600 权限）
└── sessions/                   # 会话 JSON
    └── <session_key>.json
```

**现状分析**：
- ✅ 有 Home 目录概念
- ✅ 配置文件有权限保护
- ❌ 没有记忆系统（MEMORY.md / USER.md）
- ❌ 没有技能系统
- ❌ 没有工作区上下文（`.tyagent.md`）
- ❌ 系统提示词完全硬编码在 `agent.py` 中
- ❌ 没有 `.env` 分离敏感信息
- ❌ 没有 Profile 模式
- ❌ 没有缓存分层管理

---

## 二、关键设计决策的深度分析

### 2.1 "要不要 Workspace 概念？"

OpenClaw 的 Workspace 是显式的（`~/.openclaw/<workspace>/`），Hermes 改成了隐式的（`.hermes.md` 文件在项目目录中）。

**Hermes 的设计哲学**：
> 用户已经在用文件系统组织项目了，不要在 Home 目录里再建一套项目体系。Agent 的"工作上下文"就是当前工作目录。

这个设计非常精妙：
- 你在 `~/project/tyagent/` 下工作，`.hermes.md` 就在这个目录
- 你在 `~/research/thesis/` 下工作，另一个 `.hermes.md` 就在那里
- 向上遍历到 git root，防止在子目录里遗漏

**对 tyagent 的启示**：

应该引入**"当前工作目录敏感"**的上下文机制。当用户通过飞书让 tyagent 执行文件操作时，agent 应该知道自己"现在的工作目录"是什么。

但飞书场景和 CLI 场景不同：
- CLI：`cd` 到哪里，工作目录就是哪里
- 飞书：用户可能在聊天中随时切换上下文

所以建议：
1. 引入 `.tyagent.md` 工作区上下文文件（类似 `.hermes.md`）
2. 但默认工作目录应该是**会话级别**的（每个会话可以有自己的工作目录）
3. 在 config 中增加 `default_working_dir` 配置

### 2.2 "记忆放在哪里？"

Hermes 将记忆放在 `~/.hermes/memories/` 下，有两个文件：

**MEMORY.md — Agent 的个人笔记**
- 记录工具怪癖、环境事实、API 坑点
- "Kimi Coding API User-Agent 必须是 `KimiCLI/1.30.0`"
- "lark-oapi v2 builder API 已变更，`.body()` → `.request_body()`"
- 这些事实会注入系统提示词，让 agent 不再重复踩坑

**USER.md — 用户画像**
- 记录用户偏好、习惯、禁忌
- "用户偏好简洁回答"
- "用户是 PhD 学生，研究领域是复杂系统"
- 这些会帮助 agent 调整沟通风格

**设计约束**：
- 字符限制而非 token 限制（模型无关）
- 快照模式：系统提示词中的记忆是启动时的快照，运行时写入不更新快照（保持 prefix cache 稳定）
- 工具调用后才能写入（agent 主动决定什么值得记）

**对 tyagent 的启示**：

必须引入记忆系统。tyagent 现在每次重启 gateway 后，之前会话中 agent 学到的所有事实都丢失了。这非常浪费。

但考虑到 tyagent 的定位是"轻量级"，不应该照搬 hermes 的完整实现。建议：

1. **简化版记忆**：`~/.tyagent/memories/`
   - `AGENT.md` — agent 学到的环境事实（合并 MEMORY.md 的功能）
   - `USER.md` — 用户画像
   - 每个文件上限 2000 字符（比 hermes 稍大，因为 tyagent 目前只有这一个记忆渠道）
   - 使用 `§` 分隔符（与 hermes 兼容，方便未来迁移）

2. **不引入完整的 MemoryManager 架构**
   - hermes 的 MemoryManager 支持插件式记忆提供者（Honcho、Hindsight、Mem0 等）
   - tyagent 第一阶段只需要内置的文件记忆就够了

### 2.3 "提示词放在哪里？"

Hermes 的系统提示词是**代码组装 + 上下文文件注入**：

```
系统提示词 = 代码硬编码块
          + MEMORY.md 快照
          + USER.md 快照
          + .hermes.md（工作区上下文）
          + 技能索引
          + 平台特定提示
```

这种设计的优点是：
- 核心指导原则在代码中（版本控制、统一更新）
- 动态内容在文件中（用户可自定义、持久化）
- 工作区上下文在项目中（随项目走）

**对 tyagent 的启示**：

tyagent 当前系统提示词完全硬编码在 `agent.py` 中：
```python
system_prompt="You are a helpful assistant."
```

这太简陋了。建议分层：

1. **代码层**：基础身份 + 核心指导（tool use enforcement、记忆使用指导等）
2. **文件层**：`~/.tyagent/prompts/system.md` — 用户可覆盖的系统提示词
3. **工作区层**：`.tyagent.md` — 项目特定的上下文
4. **记忆层**：`AGENT.md` + `USER.md` — 持久化的事实

### 2.4 "凭证怎么管理？"

Hermes 的方案：
- `config.yaml`：非敏感配置（模型选择、平台开关、行为参数）
- `.env`：敏感凭证（API key、token、密码）
- `.env` 通过 `python-dotenv` 加载到环境变量

这样：
- `config.yaml` 可以进 git（脱敏后）
- `.env` 绝不进 git（在 `.gitignore` 中）
- 符合 12-factor app 原则

**对 tyagent 的启示**：

tyagent 当前把所有东西都放在 `config.yaml` 中（包括 `api_key`），虽然加了 0o600 权限，但：
- 不方便版本控制
- 不方便多环境切换
- 一旦泄露就是全部泄露

建议：
1. 引入 `.env` 文件支持
2. `config.yaml` 中的 `api_key`、`app_secret` 等改为从环境变量读取
3. `.env` 文件由 tyagent 自动创建和管理

### 2.5 "Profile 模式要不要？"

Hermes 支持 Profile：
```
~/.hermes/profiles/
├── coder/           # 编程助手
├── writer/          # 写作助手
└── default/         # 默认
```

切换：`hermes --profile coder`

每个 Profile 有独立的：
- config.yaml
- .env
- memories/
- skills/
- home/（子进程隔离）

**对 tyagent 的启示**：

Profile 对 tyagent 来说是** nice-to-have，不是 must-have**。因为：
- tyagent 目前主要走飞书gateway路线，用户场景相对单一
- Profile 增加了复杂性
- 但未来的扩展性需要考虑

建议：
1. 第一阶段不实现 Profile
2. 但在目录结构上预留空间（`~/.tyagent/` 作为 root，`profiles/` 子目录未来可加）
3. `home_dir` 配置已经存在，只是目前默认指向 root

---

## 三、给 tyagent 的建议设计

### 3.1 目标目录结构

```
~/.tyagent/                    # Home（可自定义 TY_AGENT_HOME）
│
├── config.yaml                 # 主配置（非敏感）
├── .env                        # 敏感凭证（0o600，.gitignore）
│
├── memories/                   # 持久记忆
│   ├── AGENT.md                # Agent 学到的环境事实
│   └── USER.md                 # 用户画像与偏好
│
├── prompts/                    # 提示词模板
│   └── system.md               # 用户可覆盖的系统提示词
│
├── skills/                     # 技能（预留目录）
│
├── sessions/                   # 会话数据
│   └── <session_key>.json
│
├── cache/                      # 缓存
│   ├── feishu/                 # 飞书媒体缓存
│   └── images/                 # 通用图片缓存
│
└── logs/                       # 运行日志
    └── gateway/
        └── <date>.log
```

### 3.2 系统提示词组装流程

```
系统提示词 = [基础身份]                    ← 代码硬编码
          + [核心指导原则]                ← 代码硬编码（tool use、记忆使用等）
          + [平台特定提示]                ← 根据当前平台动态选择
          + [AGENT.md 内容]              ← 从文件读取（启动时快照）
          + [USER.md 内容]               ← 从文件读取（启动时快照）
          + [~/.tyagent/prompts/system.md]  ← 用户自定义（可选）
          + [.tyagent.md]               ← 工作区上下文（可选，向上遍历到 git root）
```

### 3.3 与 hermes 的差异点（有意的简化）

| 功能 | hermes | tyagent（建议） | 理由 |
|------|--------|----------------|------|
| 记忆上限 | MEMORY 2200 + USER 1375 | 各 2000 | 单通道，稍大 |
| 记忆提供者插件 | 支持 Honcho/Mem0 等 | 仅内置 | 保持轻量 |
| 技能系统 | 完整（70+ 内置） | 预留目录 | 未来扩展 |
| Profile | 完整支持 | 预留结构 | 未来扩展 |
| 上下文压缩 | 有 | 无 | 轻量定位 |
| 子进程 HOME 隔离 | 有 | 无 | 暂不涉及复杂工具链 |
| 工作区上下文 | `.hermes.md` | `.tyagent.md` | 同名替换 |
| 记忆工具 | 独立 tool schema | 通过现有工具调用 | 简化工具系统 |

### 3.4 实施优先级

**P0（立即做）**：
1. 创建 `memories/` 目录和 `AGENT.md` + `USER.md`
2. 在 `agent.py` 中读取并注入记忆到系统提示词
3. 提供 `memory` 工具（add/replace/remove）

**P1（近期做）**：
4. 引入 `.env` 分离敏感信息
5. 创建 `prompts/system.md` 用户自定义提示词
6. 引入 `.tyagent.md` 工作区上下文

**P2（未来做）**：
7. `skills/` 目录和基础技能加载
8. `logs/` 结构化日志
9. Profile 模式

---

## 四、一个具体的例子

假设用户通过飞书跟 tyagent 聊天：

**第一轮对话**：
用户："帮我在 project 文件夹下用 uv 创建一个 tyagent 项目"
→ Agent 执行后，发现 uv 默认使用 agent 的 home 目录而非用户 home
→ Agent 通过 memory 工具写入：
```
§
uv init 创建的默认 ~ 指向 agent 的 profile home（/home/wtyopenclaw/.hermes/profiles/kimihermes/home），
不是用户的实际 home（/home/wtyopenclaw）。使用 uv init 时需显式指定绝对路径。
```

**第二轮对话**（重启 gateway 后）：
用户："再用 uv 创建个项目"
→ Agent 在系统提示词中看到记忆："uv init 的默认 ~ 指向 agent 的 profile home..."
→ Agent 直接问："你想在哪个绝对路径下创建？"
→ 不再重复踩坑

这就是工作空间/记忆系统的价值——**让 agent 的跨会话经验真正积累下来**。

---

*文档完成。以上分析基于对 hermes-agent 源码（尤其是 `hermes_constants.py`、`tools/memory_tool.py`、`agent/prompt_builder.py`、`agent/memory_provider.py`、`hermes_cli/claw.py`）的直接阅读。*
