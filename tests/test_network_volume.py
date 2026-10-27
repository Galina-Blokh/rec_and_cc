"""Network-volume detection for output_dir (SMB / AFP / NFS / WebDAV, UNC, mapped drives)
and the one-time warning, mirroring the cloud-sync one. Platform calls are faked.

Stories: US-34, US-14.
"""

from __future__ import annotations

import logging
import queue
import subprocess
import sys
import threading

import pytest

import recandcc.cloudsync as cs
import recandcc.ui.common as common
from recandcc.config import Config
from recandcc.ui.common import Controller

LOCAL = 0x1000


@pytest.fixture(autouse=True)
def _no_realpath(monkeypatch):
    """Fake paths (UNC, /mnt/...) must not be resolved against the host filesystem."""
    monkeypatch.setattr(cs.os.path, "realpath", lambda p, *a, **kw: str(p))


def test_links_to_shares_are_resolved(tmp_path, monkeypatch):
    monkeypatch.setattr(cs.os.path, "realpath", lambda p, *a, **kw: "/mnt/nas/rec")
    mounts = "srv:/x /mnt/nas nfs rw 0 0\n"
    assert cs.linux_network_volume(tmp_path / "link", mounts) == "NFS"


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
    assert cs.windows_network_volume(r"\\?\Z:\Rec", drive_type=lambda _r: 4) == "a network drive"
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
    res = cs.network_volume(tmp_path, platform="darwin")
    assert res is cs.PROBE_FAILED and not res  # distinct from "not a network volume" (None)


def test_fuse_mounts_are_judged_by_their_source():
    def fuse(t, src, flags=0):
        return lambda p: (t, flags, src)

    nv = cs.mac_network_volume
    # network FUSE keeps the privacy warning
    assert nv("/V/x", fuse("macfuse", "me@host:/data")) == "a network volume"
    assert nv("/V/x", fuse("fuse-t", "remote:bucket")) == "a network volume"
    assert nv("/V/x", fuse("osxfuse", "//nas/share")) == "a network volume"
    assert nv("/V/x", fuse("macfuse", "rclone")) == "a network volume"
    assert nv("/V/x", fuse("macfuse", "smbnetfs")) == "a network volume"
    assert nv("/V/x", fuse("macfuse", "mystery")) == "a network volume"  # unsure: warn
    # known local vaults do not
    assert nv("/V/x", fuse("macfuse", "Cryptomator@/Users/me/vault")) is None
    assert nv("/V/x", fuse("fuse-t", "gocryptfs")) is None
    assert nv("/V/x", fuse("fuse-t", "/Users/me/vault.gocryptfs")) is None
    # a vault name inside a network source's path does not make it local
    assert nv("/V/x", fuse("fuse-t", "me@host:/backups/gocryptfs")) == "a network volume"
    assert nv("/V/x", fuse("macfuse", "//nas/cryptomator")) == "a network volume"
    assert "cryptomator-" not in cs._LOCAL_FUSE
    # local vault paths with vault names in any component, or look-alike words, stay local
    assert nv("/V/x", fuse("macfuse", "/Users/me/cryfs/vault1")) is None
    assert nv("/V/x", fuse("macfuse", "/Users/me/Cryptomator/vaultX")) is None
    assert nv("/V/x", fuse("macfuse", "/Users/me/rclone-vault/gocryptfs")) is None
    assert nv("/V/x", fuse("macfuse", "/Volumes/rclone-cache/x.gocryptfs")) is None
    # a vault word buried inside another word is not a vault
    assert nv("/V/x", fuse("macfuse", "/Users/me/notgocryptfsx/y")) == "a network volume"
    assert nv("/V/x", fuse("macfuse", "/Users/me/my-cryfs-stuff/y")) == "a network volume"
    # network names are case-insensitive and anchored to the fstype or source head
    assert nv("/V/x", fuse("macfuse", "Rclone")) == "a network volume"
    assert nv("/V/x", fuse("macfuse", "SSHFS#me@h")) == "a network volume"
    assert nv("/V/x", fuse("s3fs", "cryptomator-backup")) == "a network volume"
    assert nv("/V/x", fuse("gcsfuse", "gocryptfs-bucket")) == "a network volume"
    assert nv("/V/x", fuse("macfuse", "s3fs")) == "a network volume"
    # vault names are not a bare substring of an odd non-path source
    assert nv("/V/x", fuse("macfuse", "backup-gocryptfs")) == "a network volume"
    assert nv("/V/x", fuse("macfuse", "veracrypt")) is None
    assert nv("/V/x", lambda p: ("smbfs", cs._MNT_LOCAL)) == "SMB"
    mounts = "x /mnt/vault fuse.cryptomator rw 0 0\ns:/ /mnt/s fuse.sshfs rw 0 0\n"
    assert cs.linux_network_volume("/mnt/vault/a", mounts) is None
    assert cs.linux_network_volume("/mnt/s/a", mounts) == "SSHFS"


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


def test_a_hung_probe_never_stalls_the_refresh(tmp_path, monkeypatch):
    release = threading.Event()
    monkeypatch.setattr(common, "NET_PROBE_WAIT_S", 0.05)
    monkeypatch.setattr(common, "network_volume", lambda p: release.wait(10) and "NFS")
    ctrl = Controller(
        Config(output_dir=tmp_path),
        monitor=_Monitor(),
        recorder=object(),
        tq=_Q(),
        start_threads=False,
    )
    ctrl._check_network_volume()  # returns after the short wait although the probe hangs
    ctrl._check_network_volume()  # does not start a second stuck probe
    assert _notices(ctrl) == [] and ctrl._net_warned == set()
    release.set()
    ctrl._net_probes[tmp_path].join(5)
    assert len(_notices(ctrl)) == 1  # a late answer still warns, once


def test_cloud_synced_folder_gets_only_the_cloud_notice(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "cloud_sync_provider", lambda p: "OneDrive")
    monkeypatch.setattr(common, "network_volume", lambda p: "a network share")
    ctrl = Controller(
        Config(output_dir=tmp_path),
        monitor=_Monitor(),
        recorder=object(),
        tq=_Q(),
        start_threads=False,
    )
    ctrl._check_network_volume()
    assert _notices(ctrl) == []


def test_a_failed_or_hung_probe_is_not_retried_every_refresh(tmp_path, monkeypatch):
    calls = []

    def boom(p):
        calls.append(p)
        raise OSError("stalled")

    monkeypatch.setattr(common, "network_volume", boom)
    ctrl = Controller(
        Config(output_dir=tmp_path),
        monitor=_Monitor(),
        recorder=object(),
        tq=_Q(),
        start_threads=False,
    )
    ctrl._check_network_volume()
    ctrl._net_probes[tmp_path].join(5)
    ctrl._check_network_volume()
    assert len(calls) == 1
    ctrl._net_retry[tmp_path] = 0.0  # the retry interval has passed
    ctrl._check_network_volume()
    assert len(calls) == 2


def test_a_hung_probe_does_not_block_other_folders(tmp_path, monkeypatch):
    release = threading.Event()
    monkeypatch.setattr(common, "NET_PROBE_WAIT_S", 0.05)

    def probe(p):
        return (release.wait(10) and "NFS") if p == tmp_path / "dead" else "SMB"

    monkeypatch.setattr(common, "network_volume", probe)
    cfg = Config(output_dir=tmp_path / "dead")
    ctrl = Controller(cfg, monitor=_Monitor(), recorder=object(), tq=_Q(), start_threads=False)
    ctrl._check_network_volume()  # hangs
    cfg.output_dir = tmp_path / "healthy"
    ctrl._check_network_volume()
    ctrl._net_probes[tmp_path / "healthy"].join(5)
    assert len(_notices(ctrl)) == 1
    release.set()
    ctrl._net_probes[tmp_path / "dead"].join(5)


def _ctrl(tmp_path):
    return Controller(
        Config(output_dir=tmp_path),
        monitor=_Monitor(),
        recorder=object(),
        tq=_Q(),
        start_threads=False,
    )


def test_local_result_is_cached_per_path(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(common, "network_volume", lambda p: calls.append(p))
    ctrl = _ctrl(tmp_path)
    ctrl._check_network_volume()
    ctrl._net_probes[tmp_path].join(5)
    ctrl._check_network_volume()
    assert len(calls) == 1 and tmp_path not in ctrl._net_retry


def test_cloud_synced_folder_is_cached_too(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(common, "cloud_sync_provider", lambda p: calls.append(p) or "OneDrive")
    ctrl = _ctrl(tmp_path)
    ctrl._check_network_volume()
    ctrl._net_probes[tmp_path].join(5)
    ctrl._check_network_volume()
    assert len(calls) == 1


def test_probe_threads_are_capped(tmp_path, monkeypatch):
    release = threading.Event()
    monkeypatch.setattr(common, "NET_PROBE_WAIT_S", 0.01)
    monkeypatch.setattr(common, "NET_PROBE_MAX_THREADS", 2)
    monkeypatch.setattr(common, "network_volume", lambda p: release.wait(10) and None)
    cfg = Config(output_dir=tmp_path)
    ctrl = Controller(cfg, monitor=_Monitor(), recorder=object(), tq=_Q(), start_threads=False)
    for i in range(5):
        cfg.output_dir = tmp_path / str(i)
        ctrl._check_network_volume()
    assert len(ctrl._net_probes) == 2
    release.set()


def test_probe_failed_result_backs_off_like_an_error(tmp_path, monkeypatch):
    calls = []

    def failed(p):
        calls.append(p)
        return cs.PROBE_FAILED

    monkeypatch.setattr(common, "network_volume", failed)
    ctrl = _ctrl(tmp_path)
    ctrl._check_network_volume()
    ctrl._net_probes[tmp_path].join(5)
    ctrl._check_network_volume()
    assert len(calls) == 1 and _notices(ctrl) == []  # backed off, no notice
    ctrl._net_retry[tmp_path] = 0.0
    ctrl._check_network_volume()
    ctrl._net_probes[tmp_path].join(5)
    assert len(calls) == 2


def test_local_cache_expires_after_the_ttl(tmp_path, monkeypatch):
    calls = []
    now = [1000.0]
    monkeypatch.setattr(common.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(common, "network_volume", lambda p: calls.append(p))
    ctrl = _ctrl(tmp_path)
    ctrl._check_network_volume()
    ctrl._net_probes[tmp_path].join(5)
    now[0] += common.NET_LOCAL_TTL_S - 1
    ctrl._check_network_volume()
    assert len(calls) == 1
    now[0] += 2
    ctrl._check_network_volume()
    ctrl._net_probes[tmp_path].join(5)
    assert len(calls) == 2
