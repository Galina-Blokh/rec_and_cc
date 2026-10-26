// `rcc-audiotap monitor`: periodically report which processes are doing audio I/O.

import AppKit
import CoreAudio
import Darwin
import Foundation

struct AudioProcessInfo: Encodable {
    let pid: Int32
    let bundle_id: String
    let name: String
    let input: Bool
    let output: Bool
}

/// Snapshot of every Core Audio process object whose input or output is currently running.
///
/// The HAL keeps one "process object" per client process that has ever used Core Audio.
/// kAudioProcessPropertyIsRunningInput/Output are true while that process has an IO cycle
/// running in the corresponding direction (i.e. is actually recording / playing right now).
func activeAudioProcesses() throws -> [AudioProcessInfo] {
    let objects = try readObjectList(AudioObjectID(kAudioObjectSystemObject),
                                     kAudioHardwarePropertyProcessObjectList,
                                     name: "process object list")
    var result: [AudioProcessInfo] = []
    for object in objects {
        // A process can exit between listing and querying; skip it instead of failing.
        do {
            let input = try readUInt32(object, kAudioProcessPropertyIsRunningInput, name: "IsRunningInput") != 0
            let output = try readUInt32(object, kAudioProcessPropertyIsRunningOutput, name: "IsRunningOutput") != 0
            guard input || output else { continue }
            let pid = try readValue(object, kAudioProcessPropertyPID, as: pid_t.self, initial: -1, name: "PID")
            let bundleID = (try? readString(object, kAudioProcessPropertyBundleID, name: "BundleID")) ?? nil
            result.append(AudioProcessInfo(
                pid: pid,
                bundle_id: bundleID ?? "",
                name: displayName(pid: pid, bundleID: bundleID),
                input: input,
                output: output))
        } catch {
            continue
        }
    }
    return result
}

/// NSRunningApplication name, else the executable name, else the bundle id.
func displayName(pid: pid_t, bundleID: String?) -> String {
    if let name = NSRunningApplication(processIdentifier: pid)?.localizedName, !name.isEmpty {
        return name
    }
    var buffer = [CChar](repeating: 0, count: 256)
    if proc_name(pid, &buffer, UInt32(buffer.count)) > 0 {
        let name = String(cString: buffer)
        if !name.isEmpty { return name }
    }
    if let bundleID, !bundleID.isEmpty { return bundleID }
    return "pid \(pid)"
}

func runMonitor(interval: Double) throws -> Never {
    try requireProcessTapSupport()
    signal(SIGPIPE, SIG_IGN)
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
    while true {
        let line: Data
        do {
            line = try encoder.encode(try activeAudioProcesses())
        } catch {
            logError("\(error)")
            line = Data("[]".utf8)
        }
        guard writeAll(STDOUT_FILENO, line + Data("\n".utf8)) else {
            exit(0)  // Reader went away: nothing more to do.
        }
        Thread.sleep(forTimeInterval: interval)
    }
}

/// write(2) the whole buffer, retrying on EINTR / partial writes. False on EPIPE or other errors.
func writeAll(_ fd: Int32, _ data: Data) -> Bool {
    data.withUnsafeBytes { raw -> Bool in
        guard var ptr = raw.baseAddress else { return true }
        var remaining = raw.count
        while remaining > 0 {
            let n = write(fd, ptr, remaining)
            if n < 0 {
                if errno == EINTR { continue }
                return false
            }
            ptr += n
            remaining -= n
        }
        return true
    }
}
