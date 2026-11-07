from __future__ import annotations

from memory import __main__ as cli
from memory import store


def test_restore_pin_and_listing(memory_db, capsys):
    fid = store.add_fact("likes pizza")
    store.forget_facts([fid])

    cli.main(["--facts", "--all"])
    assert "forgotten" in capsys.readouterr().out

    cli.main(["--restore", str(fid), "--pin", str(fid)])
    out = capsys.readouterr().out
    assert "restored" in out and "pinned" in out
    assert store.get_all_facts() == ["likes pizza"] and store.get_fact(fid)["pinned"]

    cli.main(["--events", "10"])
    shown = capsys.readouterr().out
    assert "restore" in shown and "pin" in shown


def test_restoring_something_live_says_so(memory_db, capsys):
    fid = store.add_fact("likes pizza")
    cli.main(["--restore", str(fid)])
    assert "nothing to restore" in capsys.readouterr().out
