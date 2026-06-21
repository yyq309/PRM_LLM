# VulnHub / Real-Target Correspondence

This document grounds the abstract WebAttackSim templates in real single-host Web
penetration patterns, following method §12.1 (templates should be reverse-abstracted
from real attack records, not invented). It is the documentation-level correspondence
that the §14 Docker/VulnHub adapter would later validate empirically. It is **not** a
claim of byte-for-byte reproduction of any specific box — the correspondence is at the
level of the *attack chain* (preconditions → effects → reward events) and the
*vulnerability family*, which is what the value oracle and PRM actually learn.

For each abstract family we give: the OWASP/CWE label, the abstract action chain (the
16-action schema), and representative public boxes / labs whose published walkthroughs
follow the same chain. Box names are illustrative of the *pattern*; many boxes mix
several patterns.

## Mapping

| Abstract family (difficulty) | OWASP / CWE | Abstract action chain | Representative real boxes / labs (pattern) |
|---|---|---|---|
| `backup_or_config_leak` → flag (easy) | A05:2021 Security Misconfiguration / CWE-530 Exposed Backup, CWE-538 | web_path_enumeration → content_retrieval → (goal) | DVWA file-disclosure, many VulnHub "easy" boxes exposing `/backup`, `.bak`, `.swp`, `/.env` |
| `backup_or_config_leak` → credential → login (medium) | A05 / CWE-522 Insufficiently Protected Credentials | path_enum → content_retrieval → input_discovery → credential_use → sensitive_file_read | VulnHub **Basic Pentesting 1** (exposed config → creds), `/.git` source leak boxes |
| `default_or_weak_password` → admin (easy) | A07:2021 Identification & Auth Failures / CWE-521, CWE-798 | path_enum → input_discovery → auth_attempt → sensitive_file_read | VulnHub **Mr-Robot** (weak WP creds), **Basic Pentesting**, default-credential CMS boxes |
| `sqli` → dump credential → login (medium) | A03:2021 Injection / CWE-89 | path_enum → input_discovery → vulnerability_check → exploit_attempt → credential_use → sensitive_file_read | VulnHub **SQLi-to-Shell**, PortSwigger SQLi labs (login bypass / UNION dump), **DC-1**-style |
| `lfi` → read config / dump credential → login (medium) | A03 / CWE-22 Path Traversal, CWE-98 | path_enum → input_discovery → vulnerability_check → exploit_attempt → credential_use → sensitive_file_read | VulnHub LFI boxes (`?page=`, `?file=`), PortSwigger path-traversal labs |
| `rce` → web shell → read flag (hard) | A03 / CWE-78 OS Command Injection, CWE-94 | path_enum → input_discovery → vulnerability_check → exploit_attempt → command_execution → sensitive_file_read | VulnHub command-injection boxes (`?cmd=`, `?host=` ping), **Kioptrix**-style services |
| `file_upload` → web shell → read flag (hard) | A04/A05 / CWE-434 Unrestricted Upload | (auth) → file_upload_attempt → command_execution → sensitive_file_read | VulnHub **SickOs**, upload-bypass boxes, DVWA file-upload |
| `sqli`/`leak` → upload → web shell (hard) | CWE-89 + CWE-434 chained | injection/leak → credential_use → file_upload_attempt → command_execution | Multi-stage VulnHub boxes (creds → admin panel → upload) |
| `*` → web shell → local privilege escalation → root (hard) | A04 Insecure Design / CWE-269, CWE-250 (SUID/sudo) | … → command_execution → privilege_escalation → sensitive_file_read | VulnHub **Kioptrix**, **DC-1..9**, **Lin.Security** (SUID/sudo/cron privesc to root) |
| `authed_injection` → web shell → flag (hard) | A07 + A03 / CWE-862 Missing Authorization + CWE-78 | path_enum → input_discovery → **auth_attempt** → path_enum → input_discovery → vulnerability_check → exploit_attempt → command_execution → sensitive_file_read | VulnHub boxes whose injectable/command feature lives **behind login** (admin-panel RCE), **DC-2**-style authed exec |
| `chained_exploit` → web shell → flag (hard) | A03 chained / CWE-89 → CWE-78 | path_enum → input_discovery → vuln_check(v1) → exploit(v1→creds) → **vuln_check(v2 requires v1)** → exploit(v2→shell) → command_execution → sensitive_file_read | Multi-stage injection boxes where SQLi/info-disclosure **unlocks** a second exploit (chained CVE), **DC-1**/**Raven**-style |
| `leak_authed_privesc` → web shell → root (hard) | A05 + A07 + A03 + A04 / CWE-522 → CWE-78 → CWE-250 | path_enum → content_retrieval(leak creds) → path_enum → input_discovery → **credential_use** → path_enum → input_discovery → vuln_check(requires auth) → exploit → command_execution → **privilege_escalation** → sensitive_file_read | Full-chain VulnHub boxes (**Raven 2**, **DC-6/7**): leaked creds → authenticated RCE → SUID/sudo privesc to root |

## Frozen schema and coverage gaps

The main experiment freezes the 16-action schema (method §13.1: cross-schema results
are not comparable). The `coverage_audit.py` grid is therefore bounded by what this
schema can express. Cells that remain empty are **structurally implausible under the
current schema**, not arbitrary omissions, and are the explicit candidates for a §6.1
versioned schema extension:

- `lfi @ easy` / `lfi @ hard`: a direct file-read LFI needs a `file` vulnerability
  effect (the schema only models `credentials` / `shell` effects today).
- `rce @ easy|medium`, `file_upload @ easy|medium`: RCE and upload inherently yield a
  shell foothold, which this difficulty model classifies as hard.
- `host_privilege_escalation @ easy|medium`: privilege escalation requires a prior
  shell foothold, so it is hard by construction.
- `sqli @ easy`: SQLi here dumps a credential that still requires a login step (medium).

Per method §12.1, this limitation is reported rather than hidden: the real-target
`out-of-abstraction rate` (measured once the §14 adapter exists) is the empirical
measure of "how much the abstraction misses", and these empty cells are the prioritized
list for the next schema version.

## How this is used

1. The abstract chains above are what the RL value oracle is trained on; `Q_web`,
   `V_web`, and `value_gap` are defined over these chains.
2. The §14 adapter (`phi`/`psi`/`eta`) maps concrete tool output from these real boxes
   back onto the same `AbstractWebState` / `AbstractWebFeedback` / `AbstractWebAction`,
   so the PRM scores real candidate actions with the value prior learned here.
3. `adapter mapping accuracy` and `out-of-abstraction rate` on these boxes are the
   first-class real-validation metrics (method §14), reported alongside PRM uplift.
