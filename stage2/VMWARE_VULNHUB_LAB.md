# VMware VulnHub Lab

这份 runbook 负责把 `Docker/Vulhub` 之外的“整机 VM 靶场层”接到当前 `Stage 2` 工程里，重点补齐主机内提权链路。

## 推荐编排

默认三台：

| Box | 角色 | 备注 |
|---|---|---|
| `DC-1` | Web -> 主机内提权 | 适合补 Drupal 容器链缺失的主机提权层 |
| `Raven-2` | Web -> 服务/数据库提权 | 适合覆盖 MySQL/UDF 这一类服务侧提权 |
| `Toppo-1` | Web -> 本地二进制滥用提权 | 保留原始 shortlist；公开 write-up 更常把它归到 `SUID/GTFOBins`，不算特别干净的 `sudo-only` 样本 |

可替代第三台：

| Box | 用途 | 注意 |
|---|---|---|
| `Symfonos-1` | 替代型第三台 | 官方页面写明可能需要 `symfonos.local` hosts 映射；公开 write-up 常见 root 路径更接近 `PATH hijack / SUID curl` |

如果你的研究目标是“严格拉开三种主机提权向量”，建议把第三台理解成“本地二进制滥用”而不是“纯 sudo GTFOBins”。当前仓库里如果你后面想换成更干净的 `sudo` 型样本，可以再接别的 box。

## VM 导入

### 1. VMware 网络

- 新建一个 `host-only` 网段，例如 `VMnet2 / 192.168.56.0/24`
- Kali 攻击机也挂到同一个 `host-only` 网段
- 靶机只保留 **一个** 虚拟网卡
- Live 试验时不要挂 `bridged` 或 `NAT`

### 2. 导入格式

- `DC-1`: VulnHub 页面标注为 `VirtualBox OVA`，默认 `DHCP`
- `Raven-2`: VulnHub 页面标注为 `VirtualBox OVA`，默认 `DHCP`
- `Toppo-1`: VulnHub 页面说明 zip 内只有 `vmdk`，需要手工新建 VM 再挂磁盘
- `Symfonos-1`: VulnHub 页面标注为 `VirtualBox OVA`，作者说明 `VMware Workstation 15 Pro` 测过，并提示 `symfonos.local`

### 3. 快照

每台机至少保留两个快照：

- `clean-boot`: 第一次成功开机、拿到 IP、确认 Web 正常后
- `pre-run`: 每次正式试验前

## 本仓库里的配置文件

- 注册表: `WebAttackSim/stage2/vm_target_registry.json`
- VMware helper: `WebAttackSim/stage2/vm_lab.py`
- 目标 descriptor:
  - `WebAttackSim/stage2/targets/vulnhub-dc-1.json`
  - `WebAttackSim/stage2/targets/vulnhub-raven-2.json`
  - `WebAttackSim/stage2/targets/vulnhub-toppo-1.json`
  - `WebAttackSim/stage2/targets/vulnhub-symfonos-1.json`

`vm_target_registry.json` 里的 `enabled=true` 目标会参与：

- hosts 条目渲染
- HTTP 健康检查
- descriptor 的 `target` 同步

## 操作步骤

### 1. 先改注册表

编辑 `WebAttackSim/stage2/vm_target_registry.json`：

- 把每台机的 `ip` 改成你 VMware host-only 网段里的真实地址
- 如果你想用 `Symfonos-1` 替换 `Toppo-1`：
  - 把 `Toppo-1.enabled` 改成 `false`
  - 把 `Symfonos-1.enabled` 改成 `true`

### 2. 查看当前启用目标

```powershell
python -m stage2.vm_lab --list
```

### 3. 渲染 hosts 文件条目

```powershell
python -m stage2.vm_lab --render-hosts
```

把输出追加到：

- Windows 主机：`C:\Windows\System32\drivers\etc\hosts`
- Kali 攻击机：`/etc/hosts`

如果你用 `symfonos.local`，这一步基本是必做的。

### 4. 做基础健康检查

```powershell
python -m stage2.vm_lab --check
```

它会检查：

- IP 是否还是私网
- descriptor 文件是否存在
- `target` 是否仍然符合 Stage 2 的 lab-scope 约束
- 目标 hostname 是否能从当前系统解析
- HTTP healthcheck 是否能通

报告默认写到 `outputs/stage2_vm_lab_check.json`

### 5. 把 registry 里的 URL 同步进 descriptor

```powershell
python -m stage2.vm_lab --sync-targets
```

这一步会把每个已启用目标的 `target` 写回对应的 `stage2/targets/*.json`。

## 推荐的最小试验顺序

1. 先把 Docker-native 的 `DVWA` 或已有 `Drupal` 容器链跑通
2. 再接 `DC-1`
3. 然后 `Raven-2`
4. 最后再切第三台 `Toppo-1` 或 `Symfonos-1`

这样做的原因是：

- Web 入口验证先在低基础设施成本环境里完成
- 整机 VM 只负责补主机层提权，不把所有不确定性一次堆进来

## 官方入口页

- `DC-1`: <https://www.vulnhub.com/entry/dc-1,292/>
- `Raven-2`: <https://www.vulnhub.com/entry/raven-2,269/>
- `Toppo-1`: <https://www.vulnhub.com/entry/toppo-1,245/>
- `Symfonos-1`: <https://www.vulnhub.com/entry/symfonos-1,322/>
