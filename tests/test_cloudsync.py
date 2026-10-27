"""Cloud-sync detection for output_dir (iCloud / OneDrive / File Provider) and the warning.

Pure path logic with tmp dirs and a fake environment, so it runs on any OS.

Stories: US-14.
"""

from __future__ import annotations

import queue
import sys

import pytest

from recandcc import cloudsync
from recandcc.cloudsync import cloud_sync_provider, mac_provider, windows_provider
from recandcc.config import Config
from recandcc.ui.common import Controller


def test_mac_icloud_drive_and_cloudstorage(tmp_path):
    home = tmp_path
    icloud = home / "Library" / "Mobile Documents" / "com~apple~CloudDocs" / "Rec"
    assert mac_provider(icloud, home) == "iCloud Drive"
    cs = home / "Library" / "CloudStorage"
    assert mac_provider(cs / "OneDrive-Personal" / "Rec", home) == "OneDrive"
    assert mac_provider(cs / "Dropbox" / "x", home) == "Dropbox"
    assert mac_provider(cs / "GoogleDrive-me@example.com" / "My Drive", home) == "Google Drive"
    assert mac_provider(home / "Transcribes", home) is None
    assert mac_provider(tmp_path.parent / "elsewhere", home) is None


def test_mac_desktop_and_documents(tmp_path):
    home = tmp_path
    docs = home / "Documents" / "Transcribes"
    assert mac_provider(docs, home) is None  # iCloud "Desktop & Documents" off
    (home / "Library" / "Mobile Documents" / "com~apple~CloudDocs" / "Documents").mkdir(
        parents=True
    )
    assert mac_provider(docs, home) == "iCloud Drive (Desktop & Documents)"
    assert mac_provider(home / "Desktop" / "x", home) is None  # only Documents is synced
    assert mac_provider(home / "Transcribes", home) is None


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need admin rights on Windows")
def test_mac_resolves_symlinks(tmp_path):
    home = tmp_path / "home"
    real = home / "Library" / "Mobile Documents" / "com~apple~CloudDocs" / "Rec"
    real.mkdir(parents=True)
    link = home / "Rec"
    link.symlink_to(real)
    assert mac_provider(link, home) == "iCloud Drive"


def test_windows_onedrive_env_case_insensitive():
    env = {"OneDrive": r"C:\Users\Me\OneDrive", "OneDriveCommercial": r"D:\Work\OneDrive - Acme"}
    assert windows_provider(r"c:\users\me\onedrive\Documents\Transcribes", env) == "OneDrive"
    assert windows_provider(r"D:\WORK\OneDrive - Acme\Rec", env) == "OneDrive"
    assert windows_provider(r"C:\Users\Me\OneDrive", env) == "OneDrive"
    assert windows_provider(r"C:\Users\Me\OneDriveX\Rec", env) is None
    assert windows_provider(r"C:\Users\Me\Documents\Transcribes", env) is None
    assert windows_provider(r"C:\Users\Me\Documents", {}) is None


def test_dispatch_by_platform(tmp_path, monkeypatch):
    monkeypatch.setattr(cloudsync.Path, "home", lambda: tmp_path)
    icloud = tmp_path / "Library" / "Mobile Documents" / "x"
    assert cloud_sync_provider(icloud, platform="darwin") == "iCloud Drive"
    assert cloud_sync_provider(icloud, platform="linux") is None
    monkeypatch.setenv("OneDrive", str(tmp_path / "OneDrive"))
    assert cloud_sync_provider(tmp_path / "OneDrive" / "Rec", platform="win32") == "OneDrive"


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


def test_controller_warns_once_per_folder(tmp_path, monkeypatch, caplog):
    import recandcc.ui.common as common

    synced = {tmp_path / "cloud", tmp_path / "cloud2"}
    monkeypatch.setattr(
        common, "cloud_sync_provider", lambda p: "OneDrive" if p in synced else None
    )
    cfg = Config(output_dir=tmp_path / "cloud")
    ctrl = Controller(cfg, monitor=_Monitor(), recorder=object(), tq=_Q(), start_threads=False)
    with caplog.at_level("WARNING", logger="recandcc.ui.common"):
        ctrl._check_cloud_sync()
        ctrl._check_cloud_sync()  # same folder: no repeat
    got = _notices(ctrl)
    assert len(got) == 1 and "OneDrive" in got[0][2] and "local folder" in got[0][2]
    assert sum("synced to OneDrive" in r.getMessage() for r in caplog.records) == 1

    cfg.output_dir = tmp_path / "local"  # changed to a local folder: nothing to say
    ctrl._check_cloud_sync()
    assert _notices(ctrl) == []
    cfg.output_dir = tmp_path / "cloud2"  # changed to another synced folder: warn again
    ctrl._check_cloud_sync()
    ctrl._check_cloud_sync()
    assert len(_notices(ctrl)) == 1
