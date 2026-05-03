# tyagent Profile 目录重构计划

## 目标

将 `~/.tyagent/{config.yaml,memories/,sessions/}` 扁平结构改为
`~/.tyagent/profiles/<name>/{config.yaml,memories/,sessions/,identity.md,user.md,home/}`

默认 profile 名 `tyagent`（与命令名一致）。

## 设计决策

| 项 | 旧路径 | 新路径 |
|---|---|---|
| 配置 | `~/.tyagent/config.yaml` | `~/.tyagent/profiles/tyagent/config.yaml` |
| 记忆 | `~/.tyagent/memories/` | `~/.tyagent/profiles/tyagent/memories/` |
| 会话 | `~/.tyagent/sessions/` | `~/.tyagent/profiles/tyagent/sessions/` |
| 缓存 | `~/.tyagent/cache/` | `~/.tyagent/profiles/tyagent/cache/` |
| shell home | `~/.tyagent/home/` | `~/.tyagent/profiles/tyagent/home/` |
| 身份文件 | 无（硬编码在 prompt_builder） | `~/.tyagent/profiles/tyagent/identity.md` |
| 日志 | 无（stdout） | `~/.tyagent/logs/`（顶层共享，不属于任何 profile） |

核心原则：改一处 `default_home`，所有子路径自动跟随。代码里所有路径都是从 `config.home_dir` 派生的，不需要逐个改。

## CLI 用法

```
tyagent gateway                           # → profiles/tyagent/
tyagent gateway --profile coder           # → profiles/coder/
tyagent gateway --profile /absolute/path  # → 直接使用绝对路径
tyagent config --profile coder            # 查看 coder 配置
tyagent set-model --profile coder ...     # 设置 coder 模型
```

`--profile` 是全局参数（跟现有的 `-c/--config` 同级），所有子命令可用。

## 实施步骤

### Step 1 — 改 config.py：`default_home` 指向 profiles 子目录

改动文件：`tyagent/config.py`

- `default_home` 从 `_usr_home / ".tyagent"` 改为 `_usr_home / ".tyagent" / "profiles" / "tyagent"`
- 新增模块级变量 `DEFAULT_PROFILE_NAME = "tyagent"`
- `load_config()` 签名不变（仍接受 `config_path`，此时用显式路径）
- 新增 `load_config_for_profile(profile_name: str)` — 按 profile 名加载
- 新增 `profile_home(profile_name: str)` — 返回 `~/.tyagent/profiles/<name>/`

改动很小，因为所有子路径（sessions_dir, memories, cache, home）都从 `home_dir` 派生，自动跟随。

验证：`pytest tests/ -x -q`

### Step 2 — 改 CLI：加 `--profile` 全局参数

改动文件：`tyagent_cli.py`

- 在父 parser 上加 `--profile` 参数（与 `-c/--config` 同级）
- `cmd_gateway()` 改为接受 `--profile` 并传给 `run_gateway()`
- `run_gateway()` 签名加 `profile: Optional[str] = None`
- 其他命令（config, set-model, configure, setup-feishu, test-llm）也接受 `--profile`
- 如果同时传了 `--config` 和 `--profile`，`--config` 优先（显式路径 > profile 名）

验证：`pytest tests/ -x -q`，然后手动 `tyagent gateway --profile test-profile` 看目录是否创建

### Step 3 — 改 prompt_builder.py：读 identity.md

改动文件：`tyagent/prompt_builder.py`

- `build_system_prompt()` 加参数 `home_dir: Path`
- 如果 `home_dir / "identity.md"` 存在，读取其内容替换硬编码的 `TYAGENT_IDENTITY`
- 如果不存在，fallback 到硬编码常量（向后兼容新 profile 没有 identity.md 的情况）
- 同样处理 `user.md`（如果存在则追加到 system prompt）

改动文件：`tyagent/agent.py`
- `TyAgent` 加 `home_dir` 属性，传给 `build_system_prompt()`

改动文件：`tyagent/gateway/gateway.py`
- `_get_or_create_agent()` 创建 agent 时传入 `config.home_dir`

验证：`pytest tests/ -x -q`，然后在 `profiles/tyagent/identity.md` 写点内容重启 gateway 看效果

### Step 4 — 数据迁移

改动文件：`tyagent_cli.py` 加 `tyagent migrate` 子命令

- 检查 `~/.tyagent/config.yaml` 是否存在（旧结构标志）
- 如果存在且 `~/.tyagent/profiles/tyagent/` 不存在，自动迁移：
  - 创建 `~/.tyagent/profiles/tyagent/`
  - 移动 `config.yaml`, `memories/`, `sessions/`, `cache/`, `home/`
- 如果两边都存在，报错让用户手动处理
- 迁移后创建 `~/.tyagent/.migrated` 标记文件防止重复迁移

或者更简单：在 `run_gateway()` 启动时自动检测并迁移，不需要单独命令。如果旧 `~/.tyagent/config.yaml` 存在但 `~/.tyagent/profiles/tyagent/config.yaml` 不存在，自动做迁移。

选第二种（启动时自动迁移），更彻底——用户无感。

改动文件：`tyagent/gateway/gateway.py` 的 `run_gateway()`

- 在 `load_config()` 之前加迁移逻辑

验证：备份当前 `~/.tyagent/`，运行 gateway 看迁移是否成功，再恢复

### Step 5 — 更新测试

改动文件：`tests/` 中涉及路径的测试

- 检查哪些测试硬编码了 `~/.tyagent` 路径
- 改为使用 temp dir 或 mock `home_dir`

验证：`pytest tests/ -x -q`（442 tests 保持全绿）

### Step 6 — 盲审循环

整体盲审，关注：
- 并发安全（多个 gateway 实例用不同 profile？当前不支持，但路径隔离至少不会互相踩）
- systemd service 文件是否需要更新（目前 `ExecStart` 没传 `--profile`，默认用 `tyagent`，没问题）
- 测试覆盖完整性
- 边界情况：profile 名含特殊字符、绝对路径、不存在的 profile

## 不改的部分

- `SERVICE_NAME`、`_get_unit_path()` — 这些是 systemd 层面的，与 profile 无关
- `_get_python_path()` — 同样
- `session.py` — 只接受 `sessions_dir` 参数，不关心它在哪
- `db.py` — 只接受 db 路径
- 工具注册、事件系统 — 与路径无关

## 回退方案

如果出问题：`git revert` 这个 commit 即可。旧数据在迁移时只是移动（不是删除），可以手动移回去。

但考虑到用户说"彻底改"，建议不留回退路径——新结构就是唯一结构。迁移写 `shutil.move`（不是 copy+delete），原子性好。
