// `rcc-audiotap permissions | request <kind> | open-settings <kind>`
//
// TCC attributes these to the *responsible process*: when rcc-audiotap is started (via Python)
// from RecAndCC.app, prompts and System Settings entries say "RecAndCC"; when started from a
// terminal they are attributed to the terminal app.

import AppKit
import AVFoundation
import Foundation

enum PermissionKind: String {
    case microphone
    case systemAudio = "system_audio"
}

// MARK: - TCC SPI (system audio)
//
// There is no public API to query or request the "System Audio Recording" permission
// (kTCCServiceAudioCapture). Like Apple-sample-derived projects (e.g. AudioCap) we look up the
// private TCC functions at runtime, so a missing symbol degrades to "unknown" instead of a
// launch failure.

private typealias TCCPreflightFn = @convention(c) (CFString, CFDictionary?) -> Int
private typealias TCCRequestFn = @convention(c) (CFString, CFDictionary?, @convention(block) (Bool) -> Void) -> Void

private let tccHandle: UnsafeMutableRawPointer? =
    dlopen("/System/Library/PrivateFrameworks/TCC.framework/Versions/A/TCC", RTLD_NOW)

private func tccSymbol<T>(_ name: String, as _: T.Type) -> T? {
    guard let tccHandle, let symbol = dlsym(tccHandle, name) else { return nil }
    return unsafeBitCast(symbol, to: T.self)
}

private let audioCaptureService = "kTCCServiceAudioCapture" as CFString

/// "granted" | "denied" | "undetermined" | "unknown" (SPI missing).
func systemAudioStatus() -> String {
    guard let preflight = tccSymbol("TCCAccessPreflight", as: TCCPreflightFn.self) else { return "unknown" }
    switch preflight(audioCaptureService, nil) {
    case 0: return "granted"
    case 1: return "denied"
    default: return "undetermined"
    }
}

/// "granted" | "denied" | "undetermined" | "restricted"
func microphoneStatus() -> String {
    switch AVCaptureDevice.authorizationStatus(for: .audio) {
    case .authorized: return "granted"
    case .denied: return "denied"
    case .restricted: return "restricted"
    case .notDetermined: return "undetermined"
    @unknown default: return "undetermined"
    }
}

// MARK: - Commands

func printJSON(_ object: [String: Any]) {
    if let data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]),
       let line = String(data: data, encoding: .utf8) {
        print(line)
    } else {
        print("{}")
    }
    fflush(stdout)
}

func runPermissions() -> Never {
    printJSON(["microphone": microphoneStatus(), "system_audio": systemAudioStatus()])
    exit(0)
}

/// Spin the main run loop (TCC/AVFoundation callbacks may need it) until `done()` or timeout.
private func waitUntil(timeout: TimeInterval, _ done: () -> Bool) {
    let deadline = Date().addingTimeInterval(timeout)
    while !done() && Date() < deadline {
        RunLoop.main.run(until: Date().addingTimeInterval(0.1))
    }
}

func runRequest(_ kind: PermissionKind, timeout: TimeInterval) -> Never {
    switch kind {
    case .microphone:
        if microphoneStatus() == "undetermined" {
            let lock = NSLock()
            var answered = false
            AVCaptureDevice.requestAccess(for: .audio) { _ in
                lock.lock(); answered = true; lock.unlock()
            }
            waitUntil(timeout: timeout) { lock.lock(); defer { lock.unlock() }; return answered }
        }
        printJSON(["microphone": microphoneStatus()])

    case .systemAudio:
        if systemAudioStatus() != "granted" {
            requestSystemAudio(timeout: timeout)
        }
        printJSON(["system_audio": systemAudioStatus()])
    }
    exit(0)
}

private func requestSystemAudio(timeout: TimeInterval) {
    if let request = tccSymbol("TCCAccessRequest", as: TCCRequestFn.self) {
        let lock = NSLock()
        var answered = false
        request(audioCaptureService, nil) { _ in
            lock.lock(); answered = true; lock.unlock()
        }
        waitUntil(timeout: timeout) { lock.lock(); defer { lock.unlock() }; return answered }
        return
    }
    // No request SPI: starting IO on a process tap makes the system show the prompt itself.
    let capture = SystemAudioCapture(excludePIDs: []) { _ in }
    do {
        try capture.start()
    } catch {
        logError("could not start a process tap to trigger the permission prompt: \(error)")
        return
    }
    defer { capture.stop() }
    if systemAudioStatus() == "unknown" {
        // Can't observe the answer; give the user a moment with the prompt.
        waitUntil(timeout: min(timeout, 10)) { false }
    } else {
        waitUntil(timeout: timeout) { systemAudioStatus() != "undetermined" }
    }
}

func runOpenSettings(_ kind: PermissionKind) -> Never {
    let anchor = kind == .microphone ? "Privacy_Microphone" : "Privacy_AudioCapture"
    let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?\(anchor)")!
    let opened = NSWorkspace.shared.open(url)
    printJSON(["opened": opened, "url": url.absoluteString])
    exit(opened ? 0 : 1)
}
