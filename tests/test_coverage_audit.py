"""Tests for the coverage-audit difficulty/vuln-class classification (method §12.1)."""

from coverage_audit import difficulty, needs_shell, task_attack_surfaces, task_vuln_classes


def _leak_file():
    return {"leaks": {"/backup": {"files": ["flag"]}}, "files": {"flag": {"flag": "F"}},
            "goal": {"type": "read_file", "file": "flag"}}


def _default_pw():
    return {"credentials": {"a": {"weak": True}}, "forms": {"/login": ["u"]},
            "files": {"flag": {"requires_auth": "admin"}}, "goal": {"type": "read_file"}}


def _leak_login():
    return {"leaks": {"/backup": {"credentials": ["a"]}}, "credentials": {"a": {"weak": False}},
            "files": {"flag": {"requires_auth": "admin"}}, "goal": {"type": "read_file"}}


def _sqli():
    return {"parameters": {"/x": [{"name": "q", "vulnerability": "v"}]},
            "vulnerabilities": {"v": {"type": "sqli", "effects": {"credentials": ["a"]}}},
            "files": {"flag": {"requires_auth": "admin"}}, "goal": {"type": "read_file"}}


def _rce_shell():
    return {"vulnerabilities": {"v": {"type": "rce", "effects": {"shell": True}}},
            "files": {"flag": {"requires_shell": True}}, "goal": {"type": "read_file"}}


def _privesc():
    t = _rce_shell()
    t["privilege_escalation"] = {"available": True, "target_privilege": "root"}
    t["files"] = {"flag": {"requires_shell": True, "requires_privilege": "root"}}
    return t


def test_difficulty_classification():
    assert difficulty(_leak_file()) == "easy"      # direct leak read, no auth
    assert difficulty(_default_pw()) == "easy"     # weak credential login
    assert difficulty(_leak_login()) == "medium"   # leak yields credential -> login
    assert difficulty(_sqli()) == "medium"         # injection -> credential -> login
    assert difficulty(_rce_shell()) == "hard"      # needs a shell foothold
    assert difficulty(_privesc()) == "hard"        # privilege escalation


def test_needs_shell():
    assert needs_shell(_rce_shell())
    assert needs_shell(_privesc())
    assert not needs_shell(_leak_file())
    assert not needs_shell(_default_pw())


def test_vuln_classes():
    assert "sqli" in task_vuln_classes(_sqli())
    assert "rce" in task_vuln_classes(_rce_shell())
    assert "host_privilege_escalation" in task_vuln_classes(_privesc())
    assert "default_or_weak_password" in task_vuln_classes(_default_pw())


def test_attack_surfaces_always_include_path():
    assert "path" in task_attack_surfaces(_leak_file())
    assert "form" in task_attack_surfaces(_default_pw())
    assert "parameter" in task_attack_surfaces(_sqli())
