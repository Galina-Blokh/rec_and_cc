// RecAndCC.app launcher (Contents/MacOS/RecAndCC).
//
// macOS privacy permissions (microphone, system audio recording) are granted to the
// "responsible process" - for anything started from an app bundle, that is the app. This tiny
// launcher is what the bundle runs: it posix_spawns the project's Python as a *child* process
// so Python and every helper it starts (rcc-audiotap) are attributed to RecAndCC, not to
// Terminal or python3. It forwards termination signals and exits with the child's status.
//
// Configuration: Contents/Resources/launcher.json, written by build_app.sh:
//   {"python": "<repo>/.venv/bin/python", "args": ["-m", "recandcc", "run"], "repo": "<repo>"}
// The environment variable RCC_PYTHON overrides "python".

import Darwin
import Foundation

let bundleURL = Bundle.main.bundleURL
let logDir = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Logs/RecAndCC")
let logURL = logDir.appendingPathComponent("app.log")
let maxLogBytes = 10 * 1024 * 1024

// MARK: - Logging

/// Opens app.log for appending (rotating to app.log.1 when it gets large); returns its fd.
func openLog() -> Int32 {
    try? FileManager.default.createDirectory(at: logDir, withIntermediateDirectories: true)
    if let size = (try? FileManager.default.attributesOfItem(atPath: logURL.path))?[.size] as? Int,
       size > maxLogBytes {
        let rotated = logDir.appendingPathComponent("app.log.1")
        try? FileManager.default.removeItem(at: rotated)
        try? FileManager.default.moveItem(at: logURL, to: rotated)
    }
    let fd = open(logURL.path, O_WRONLY | O_CREAT | O_APPEND, 0o644)
    return fd >= 0 ? fd : STDERR_FILENO
}

let logFD = openLog()

func log(_ message: String) {
    let stamp = ISO8601DateFormatter().string(from: Date())
    let line = "\(stamp) [launcher] \(message)\n"
    _ = line.withCString { write(logFD, $0, strlen($0)) }
}

// MARK: - Configuration

struct LauncherConfig: Decodable {
    var python: String
    var args: [String]
    var repo: String?
}

func loadConfig() -> LauncherConfig {
    let url = bundleURL.appendingPathComponent("Contents/Resources/launcher.json")
    var config = LauncherConfig(python: "/usr/bin/python3", args: ["-m", "recandcc", "run"], repo: nil)
    do {
        config = try JSONDecoder().decode(LauncherConfig.self, from: Data(contentsOf: url))
    } catch {
        log("cannot read \(url.path): \(error); using defaults")
    }
    if let override = ProcessInfo.processInfo.environment["RCC_PYTHON"], !override.isEmpty {
        config.python = override
    }
    return config
}

// MARK: - Spawning

let forwardedSignals: [Int32] = [SIGTERM, SIGINT, SIGHUP, SIGQUIT]

func spawnChild(_ config: LauncherConfig) -> pid_t {
    var environment = ProcessInfo.processInfo.environment
    environment["RCC_APP_BUNDLE"] = bundleURL.path
    environment["PYTHONUNBUFFERED"] = "1"
    if let repo = config.repo { environment["RCC_REPO"] = repo }

    let argv = ([config.python] + config.args).map { strdup($0) } + [nil]
    let envp = environment.map { strdup("\($0.key)=\($0.value)") } + [nil]
    defer {
        argv.forEach { free($0) }
        envp.forEach { free($0) }
    }

    var actions = posix_spawn_file_actions_t(bitPattern: 0)
    posix_spawn_file_actions_init(&actions)
    defer { posix_spawn_file_actions_destroy(&actions) }
    posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0)
    posix_spawn_file_actions_adddup2(&actions, logFD, STDOUT_FILENO)
    posix_spawn_file_actions_adddup2(&actions, logFD, STDERR_FILENO)
    if let repo = config.repo {
        posix_spawn_file_actions_addchdir_np(&actions, repo)
    }

    // We ignore the forwarded signals ourselves (they are handled via dispatch sources);
    // ignored dispositions are inherited across exec, so reset them to default in the child.
    var attributes = posix_spawnattr_t(bitPattern: 0)
    posix_spawnattr_init(&attributes)
    defer { posix_spawnattr_destroy(&attributes) }
    var defaults = sigset_t()
    sigemptyset(&defaults)
    forwardedSignals.forEach { sigaddset(&defaults, $0) }
    sigaddset(&defaults, SIGPIPE)
    posix_spawnattr_setsigdefault(&attributes, &defaults)
    var mask = sigset_t()
    sigemptyset(&mask)
    posix_spawnattr_setsigmask(&attributes, &mask)
    posix_spawnattr_setflags(&attributes, Int16(POSIX_SPAWN_SETSIGDEF | POSIX_SPAWN_SETSIGMASK))

    var pid: pid_t = 0
    let status = posix_spawn(&pid, config.python, &actions, &attributes, argv, envp)
    guard status == 0 else {
        log("failed to start \(config.python): \(String(cString: strerror(status)))")
        exit(1)
    }
    return pid
}

// MARK: - Main

let config = loadConfig()
log("starting \(([config.python] + config.args).joined(separator: " ")) (bundle \(bundleURL.path))")
guard FileManager.default.isExecutableFile(atPath: config.python) else {
    log("Python interpreter not found at \(config.python). Run `uv sync` in the repo or set RCC_PYTHON.")
    exit(1)
}

signal(SIGPIPE, SIG_IGN)
forwardedSignals.forEach { signal($0, SIG_IGN) }
let child = spawnChild(config)
log("child pid \(child)")

var signalSources: [DispatchSourceSignal] = []
for sig in forwardedSignals {
    let source = DispatchSource.makeSignalSource(signal: sig, queue: .main)
    source.setEventHandler {
        log("received signal \(sig), forwarding to child")
        kill(child, sig)
    }
    source.resume()
    signalSources.append(source)
}

Thread.detachNewThread {
    var status: Int32 = 0
    while waitpid(child, &status, 0) < 0 && errno == EINTR {}
    // Decode the wait status by hand (WIFEXITED & co are macros Swift does not import).
    let termSignal = status & 0x7f
    let code: Int32 = termSignal == 0 ? (status >> 8) & 0xff : 128 + termSignal
    log(termSignal == 0 ? "child exited with status \(code)" : "child killed by signal \(termSignal)")
    exit(code)
}

withExtendedLifetime(signalSources) { dispatchMain() }
