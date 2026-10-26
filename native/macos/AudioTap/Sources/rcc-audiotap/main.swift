// rcc-audiotap: native macOS audio helper for RecAndCC.
//
//   rcc-audiotap monitor [--interval 1.0]       JSON line per interval: processes doing audio I/O
//   rcc-audiotap capture [--exclude-pid PID]... raw float32le mono 16 kHz system audio on stdout
//   rcc-audiotap permission [--seconds 1.5] [--no-sound]
//                                               check that system audio capture returns real audio
//   rcc-audiotap permissions                    JSON: TCC status of microphone / system audio
//   rcc-audiotap request microphone|system_audio [--timeout 60]
//                                               show the system prompt, print the resulting status
//   rcc-audiotap open-settings microphone|system_audio
//                                               open the matching System Settings privacy pane

import Foundation

let usage = """
usage:
  rcc-audiotap monitor [--interval SECONDS]
  rcc-audiotap capture [--exclude-pid PID ...]
  rcc-audiotap permission [--seconds SECONDS] [--no-sound]
  rcc-audiotap permissions
  rcc-audiotap request microphone|system_audio [--timeout SECONDS]
  rcc-audiotap open-settings microphone|system_audio

"""

func fail(_ message: String, code: Int32 = 2) -> Never {
    logError(message)
    FileHandle.standardError.write(Data(usage.utf8))
    exit(code)
}

var args = Array(CommandLine.arguments.dropFirst())
guard let command = args.first else { fail("missing subcommand") }
args.removeFirst()

/// Pops the value following a flag, failing with a usage message when missing/invalid.
func value<T>(after flag: String, _ parse: (String) -> T?) -> T {
    guard !args.isEmpty, let parsed = parse(args.removeFirst()) else {
        fail("\(flag) needs a valid value")
    }
    return parsed
}

do {
    switch command {
    case "monitor":
        var interval = 1.0
        while let flag = args.first {
            args.removeFirst()
            switch flag {
            case "--interval": interval = value(after: flag) { Double($0).flatMap { $0 > 0 ? $0 : nil } }
            default: fail("unknown option \(flag)")
            }
        }
        try runMonitor(interval: interval)

    case "capture":
        var exclude: [pid_t] = []
        while let flag = args.first {
            args.removeFirst()
            switch flag {
            case "--exclude-pid": exclude.append(value(after: flag) { pid_t($0) })
            default: fail("unknown option \(flag)")
            }
        }
        try runCapture(excludePIDs: exclude)

    case "permission":
        var seconds = 1.5
        var playSound = true
        while let flag = args.first {
            args.removeFirst()
            switch flag {
            case "--seconds": seconds = value(after: flag) { Double($0).flatMap { $0 > 0 ? $0 : nil } }
            case "--no-sound": playSound = false
            default: fail("unknown option \(flag)")
            }
        }
        try runPermissionCheck(seconds: seconds, playSound: playSound)

    case "permissions":
        guard args.isEmpty else { fail("permissions takes no options") }
        runPermissions()

    case "request":
        let kind = value(after: "request") { PermissionKind(rawValue: $0) }
        var timeout = 60.0
        while let flag = args.first {
            args.removeFirst()
            switch flag {
            case "--timeout": timeout = value(after: flag) { Double($0).flatMap { $0 > 0 ? $0 : nil } }
            default: fail("unknown option \(flag)")
            }
        }
        runRequest(kind, timeout: timeout)

    case "open-settings":
        let kind = value(after: "open-settings") { PermissionKind(rawValue: $0) }
        guard args.isEmpty else { fail("unexpected argument \(args[0])") }
        runOpenSettings(kind)

    case "-h", "--help", "help":
        print(usage, terminator: "")
        exit(0)

    default:
        fail("unknown subcommand \(command)")
    }
} catch {
    logError("\(error)")
    exit(1)
}
