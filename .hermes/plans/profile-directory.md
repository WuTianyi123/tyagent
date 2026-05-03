# Profile 目录结构规划

## 目标

```
~/.tyagent/
└── profiles/
    ├── tyagent/              # 默认 profile
    │   ├── config.yaml       # 从 ~/.tyagent/config.yaml 迁移
    │   ├── identity.md       # 新建
    │   ├── memories/         # 从 ~/.tyagent/memories/ 迁移
    │   │   ├── MEMORY.md
    │   │   └── USER.md
    │   ├── sessions/         # 从 ~/.tyagent/sessions/ 迁移
    │   │   └── sessions.db
    │   ├── skills/           # 新建
    │   ├── logs/             # 新建
    │   └── home/             # 从 ~/.tyagent/home/ 迁移
    └── coder/                # 未来第二个 profile
        └── ...
```

## 实现步骤

每一步都独立可测试，有明确的 Git commit 边界。任何一步出问题，回退到上一个 commit 即可。

---

### Step 1: CLI --profile 参数 + profile 目录创建

**范围**：tyagent_cli.py, tyagent/config.py

不改 gateway 逻辑，只加 CLI 参数和路径解析函数。

- `TyAgentConfig` 新增 `profile: Optional[str] = None`
- 新增 `resolve_profile_dir(profile_name: str) -> Path` → `~/.tyagent/profiles/<name>/`
- `cmd_gateway` 接受 `--profile` 参数，设到 config.profile
- 当 `--profile` 给定时：`mkdir -p profile_dir`，从 `profile_dir/config.yaml` 读配置（不存在就创建一个带默认值的并写回）
- 当 `--profile` 未给定时：走现有逻辑（从 `~/.tyagent/config.yaml` 读，完全不变）

**为什么安全**：不改现有路径。`--profile` 不传时一切照旧。传了就用新目录。

**测试**：
- `tyagent gateway` → 走老逻辑，从 `~/.tyagent/config.yaml` 读取 ✓
- `tyagent gateway --profile tyagent` → `mkdir -p ~/.tyagent/profiles/tyagent/`，创建默认 config.yaml ✓
- `--profile` 配合 `--config` 应报错互斥 ✓

---

### Step 2: gateway 使用 profile 目录派生子目录

**范围**：tyagent/gateway/gateway.py

- `run_gateway` 中：如果 `config.profile` 非空，用 `profile_dir` 作为 `config.home_dir`（覆盖掉配置文件里的 `home_dir` 字段）
- `memories_dir`、`sessions_dir` 等派生路径改用 `config.home_dir / "memories"` 而非直接读 config 字段
- profile home（子进程隔离目录）从 `config.home_dir / "home"` 创建

**为什么安全**：旧路径（profile=None）的派生逻辑不变。

**测试**：
- 无 profile 时 gateway 日志路径和之前一致 ✓
- 有 profile 时 memory 写入 `~/.tyagent/profiles/tyagent/memories/` ✓

---

### Step 3: prompt_builder 加载 identity.md

**范围**：tyagent/prompt_builder.py

- `build_system_prompt()` 首层从 `identity.md` 读取（如果存在），fallback 到 `TYAGENT_IDENTITY` 常量
- 新增 `load_identity(home_dir: Path) -> Optional[str]`，读 `home_dir / "identity.md"`
- 和现在一样缓存——不存进 `self.system_prompt`，而是作为新层拼在 prompt 里

**为什么安全**：identity.md 不存在就走常量，行为不变。

**测试**：
- identity.md 不存在 → 系统提示词第一段仍是 `You are tyagent...` ✓
- 创建 identity.md 后重启 gateway → 系统提示词第一段变成文件内容 ✓

---

### Step 4: 迁移

**范围**：磁盘操作，不改代码

- 创建 `~/.tyagent/profiles/tyagent/`（如果 step 1/2 还没创建）
- 移动 `~/.tyagent/config.yaml` → `profiles/tyagent/config.yaml`
- 移动 `~/.tyagent/memories/` → `profiles/tyagent/memories/`
- 移动 `~/.tyagent/sessions/` → `profiles/tyagent/sessions/`
- 移动 `~/.tyagent/home/` → `profiles/tyagent/home/`
- 修改迁移后的 `config.yaml`，删除 `home_dir` 字段（由 profile 派生）
- 创建 `profiles/tyagent/identity.md`
- 重启 gateway：`tyagent gateway --profile tyagent`

**为什么安全**：移动而非删除。旧文件留在原地做备份。确认 gateway 正常后再手动删旧文件。

---

### Step 5: 盲审循环

4 个 step 全部完成后，启动盲审循环检查改动一致性、边界情况、回退安全性。
