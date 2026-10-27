"""Network-volume detection for output_dir (SMB / AFP / NFS / WebDAV, UNC, mapped drives)
and the one-time warning, mirroring the cloud-sync one. Platform calls are faked.

Stories: US-34, US-14.
"""

from __future__ import annotations

import logging
import queue
import subprocess
import sys

import recandcc.cloudsync as cs
import recandcc.ui.common as common
from recandcc.config import Config
from recandcc.ui.common import Controller

LOCAL = 0x1000


def _mac(fstype, flags=LOCAL):
    return lambda _p: (fstype, flags)


def test_mac_filesystem_types(tmp_path):
    assert cs.mac_network_volume(tmp_path, _mac("apfs")) is None
    assert cs.mac_network_volume(tmp_path, _mac("hfs")) is None
    assert cs.mac_network_volume(tmp_path, _mac("smbfs", 0)) == "SMB"
    assert cs.mac_network_volume(tmp_path, _mac("afpfs", 0)) == "AFP"
    assert cs.mac_network_volume(tmp_path, _mac("nfs", 0)) == "NFS"
    assert cs.mac_network_volume(tmp_path, _mac("webdav", 0)) == "WebDAV"
    # unknown type that is not MNT_LOCAL is still treated as remote
    assert cs.mac_network_volume(tmp_path, _mac("weirdfs", 0)) == "a network volume"
    assert cs.mac_network_volume(tmp_path, _mac("weirdfs", LOCAL)) is None


def test_windows_unc_and_remote_drive():
    unc = cs.windows_network_volume(r"\\server\share\Rec", drive_type=lambda _r: 3)
    assert unc == "a network share"
    assert cs.windows_network_volume(r"\\?\UNC\server\share", drive_type=lambda _r: 3)
    assert cs.windows_network_volume(r"\\?\C:\Rec", drive_type=lambda _r: 3) is None
    seen = []

    def remote(root):
        seen.append(root)
        return 4

    assert cs.windows_network_volume(r"Z:\Rec", drive_type=remote) == "a network drive"
    assert seen == ["Z:\\"]
    assert cs.windows_network_volume(r"C:\Rec", drive_type=lambda _r: 3) is None


def test_linux_mounts():
    mounts = (
        "/dev/sda1 / ext4 rw 0 0\nsrv:/x /mnt/nas nfs4 rw 0 0\n//h/s /mnt/a\\040b cifs rw 0 0\n"
    )
    assert cs.linux_network_volume("/mnt/nas/rec", mounts) == "NFS"
    assert cs.linux_network_volume("/mnt/a b/rec", mounts) == "SMB"
    assert cs.linux_network_volume("/mnt/nasty", mounts) is None
    assert cs.linux_network_volume("/home/me", mounts) is None


def test_dispatch_and_failures_are_quiet(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "mac_network_volume", lambda p: "NFS")
    monkeypatch.setattr(cs, "windows_network_volume", lambda p: "a network share")
    assert cs.network_volume(tmp_path, platform="darwin") == "NFS"
    assert cs.network_volume(tmp_path, platform="win32") == "a network share"

    def boom(_p):
        raise OSError("stalled")

    monkeypatch.setattr(cs, "mac_network_volume", boom)
    assert cs.network_volume(tmp_path, platform="darwin") is None


def test_real_local_folder_is_not_network(tmp_path):
    assert cs.network_volume(tmp_path / "not" / "yet") is None


def test_importing_is_safe_without_windows_apis():
    # ctypes.windll must only be touched inside the win32 branch
    code = "import recandcc.cloudsync, recandcc.ui.common"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_warning_text():
    assert "SMB storage" in cs.network_warning_text("/x", "SMB")
    assert "on a network share" in cs.network_warning_text("/x", "a network share")


class _Monitor:
    def snapshot(self):
        return []


class _Q:
    def submit(self, *_a, **_k):
        pass

    def pending(self):
        return 0


def _notices(ctrl):
    out = []
    while True:
        try:
            ev = ctrl.events.get_nowait()
        except queue.Empty:
            return out
        if ev[0] == "notice":
            out.append(ev)


def test_controller_warns_once_per_folder_without_logging_the_path(tmp_path, monkeypatch, caplog):
    remote = {tmp_path / "nas", tmp_path / "nas2"}
    monkeypatch.setattr(common, "network_volume", lambda p: "SMB" if p in remote else None)
    cfg = Config(output_dir=tmp_path / "nas")
    ctrl = Controller(cfg, monitor=_Monitor(), recorder=object(), tq=_Q(), start_threads=False)
    with caplog.at_level(logging.INFO):
        ctrl._check_network_volume()
        ctrl._check_network_volume()
    got = _notices(ctrl)
    assert len(got) == 1 and "SMB" in got[0][2] and str(tmp_path / "nas") in got[0][2]
    assert len(caplog.records) == 1
    assert str(tmp_path) not in caplog.text

    cfg.output_dir = tmp_path / "local"
    ctrl._check_network_volume()
    assert _notices(ctrl) == []
    cfg.output_dir = tmp_path / "nas2"  # another share: warn again
    ctrl._check_network_volume()
    assert len(_notices(ctrl)) == 1


def test_refresh_runs_the_check_and_never_blocks(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "permission_status", lambda: {})
    monkeypatch.setattr(common, "network_volume", lambda p: "NFS")
    ctrl = Controller(
        Config(output_dir=tmp_path),
        monitor=_Monitor(),
        recorder=object(),
        tq=_Q(),
        start_threads=False,
    )
    ctrl._maybe_reload_config = lambda: None
    ctrl._refresh()
    ctrl._refresh()
    assert len(_notices(ctrl)) == 1


def test_cli_config_prints_warning(tmp_path, monkeypatch, capsys):
    import recandcc.cli as cli

    monkeypatch.setattr(cs, "network_volume", lambda p: "NFS")
    cli.cmd_config(None)
    assert "NFS storage" in capsys.readouterr().out
