// `rcc-audiotap settings --config <snapshot.json>`: a small SwiftUI settings window.
//
// The helper never touches config.toml. Python writes a JSON snapshot of the current values
// (--config), this window edits them, and one JSON line goes to stdout when it closes:
//   {"settings":"saved","values":{mode,output_dir,languages,keep_audio,live_captions,
//                                 speaker_names}}   Save was pressed
//   {"settings":"cancelled"}                          Cancel / window closed
// Python validates the values (config.validate_settings) and writes the TOML, so config
// validation lives in one place. The window shows settings only, no transcript or audio.
//
// The process runs with a regular activation policy while the window is open (so it can take
// keyboard focus and shows a Dock icon) and exits when it closes. No TCC permission is used.

import AppKit
import Foundation
import SwiftUI

/// The language checkboxes. Any other language already in the config is passed through.
let settingsLanguages: [(code: String, label: String)] = [
    ("en", "English"), ("ru", "Russian"), ("he", "Hebrew"),
]

/// Pure mapping between the snapshot/result JSON and the form state (unit-testable).
struct SettingsModel {
    var mode = "ask"
    var outputDir = ""
    var languages: Set<String> = []
    var otherLanguages: [String] = []
    var keepAudio = true
    var liveCaptions = true
    var speakerNames = false
    var hotkey = ""

    init(json: [String: Any]) {
        mode = json["mode"] as? String ?? mode
        outputDir = json["output_dir"] as? String ?? ""
        let all = json["languages"] as? [String] ?? []
        let known = Set(settingsLanguages.map { $0.code })
        languages = Set(all.filter { known.contains($0) })
        otherLanguages = all.filter { !known.contains($0) }
        keepAudio = json["keep_audio"] as? Bool ?? keepAudio
        liveCaptions = json["live_captions"] as? Bool ?? liveCaptions
        speakerNames = json["speaker_names"] as? Bool ?? speakerNames
        hotkey = json["hotkey"] as? String ?? ""
    }

    var canSave: Bool { !outputDir.isEmpty }  // no language checked = any language

    var result: [String: Any] {
        let picked = settingsLanguages.map { $0.code }.filter { languages.contains($0) }
        return [
            "mode": mode, "output_dir": outputDir, "languages": picked + otherLanguages,
            "keep_audio": keepAudio, "live_captions": liveCaptions, "speaker_names": speakerNames,
        ]
    }
}

/// A class with @Published rather than SwiftUI's @State: @State is a macro and needs the
/// SwiftUI macro plugin, which Command Line Tools alone do not ship.
final class SettingsStore: ObservableObject {
    @Published var model: SettingsModel
    init(_ model: SettingsModel) { self.model = model }
}

struct SettingsView: View {
    @ObservedObject var store: SettingsStore
    var model: SettingsModel { get { store.model } nonmutating set { store.model = newValue } }
    let finish: ([String: Any]?) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Picker("When audio is detected", selection: $store.model.mode) {
                Text("Off").tag("off")
                Text("Ask").tag("ask")
                Text("Record automatically").tag("auto")
            }
            HStack {
                Text("Transcripts folder")
                Text(model.outputDir).lineLimit(1).truncationMode(.head).foregroundStyle(.secondary)
                Spacer()
                Button("Choose…") { chooseFolder() }
            }
            HStack {
                Text("Languages (none = any)")
                ForEach(settingsLanguages, id: \.code) { lang in
                    Toggle(lang.label, isOn: Binding(
                        get: { model.languages.contains(lang.code) },
                        set: { on in
                            if on { model.languages.insert(lang.code) } else { model.languages.remove(lang.code) }
                        }))
                }
            }
            Toggle("Keep audio after transcribing", isOn: $store.model.keepAudio)
            Toggle("Show live captions", isOn: $store.model.liveCaptions)
            Toggle("Remember speakers' voices (stored on this Mac)", isOn: $store.model.speakerNames)
            HStack {
                Text("Shortcut")
                Text(model.hotkey.isEmpty ? "off" : model.hotkey).foregroundStyle(.secondary)
                Text("(change `hotkey` in config.toml)").font(.caption).foregroundStyle(.tertiary)
            }
            HStack {
                Spacer()
                Button("Cancel") { finish(nil) }.keyboardShortcut(.cancelAction)
                Button("Save") { finish(model.result) }
                    .keyboardShortcut(.defaultAction).disabled(!model.canSave)
            }
        }
        .padding(20)
        .frame(width: 520)
    }

    private func chooseFolder() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.canCreateDirectories = true
        if !model.outputDir.isEmpty { panel.directoryURL = URL(fileURLWithPath: model.outputDir) }
        if panel.runModal() == .OK, let url = panel.url { model.outputDir = url.path }
    }
}

private final class SettingsDelegate: NSObject, NSApplicationDelegate, NSWindowDelegate {
    func windowWillClose(_ notification: Notification) { emitAndExit(nil) }
    func applicationShouldTerminateAfterLastWindowClosed(_ app: NSApplication) -> Bool { true }
}

private var settingsDone = false

func emitAndExit(_ values: [String: Any]?) {
    guard !settingsDone else { return }
    settingsDone = true
    if let values {
        printJSON(["settings": "saved", "values": values])
    } else {
        printJSON(["settings": "cancelled"])
    }
    exit(0)
}

private var settingsDelegate: SettingsDelegate?

func runSettings(configPath: String) -> Never {
    guard let data = FileManager.default.contents(atPath: configPath),
          let json = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
    else {
        logError("settings: cannot read \(configPath)")
        exit(1)
    }
    let app = NSApplication.shared
    app.setActivationPolicy(.regular)  // a window needs focus; .accessory cannot take it
    let delegate = SettingsDelegate()
    settingsDelegate = delegate
    app.delegate = delegate

    let view = SettingsView(store: SettingsStore(SettingsModel(json: json)), finish: emitAndExit)
    let window = NSWindow(contentViewController: NSHostingController(rootView: view))
    window.title = "RecAndCC Settings"
    window.styleMask = NSWindow.StyleMask([.titled, .closable])
    window.delegate = delegate
    window.center()
    window.makeKeyAndOrderFront(nil)
    app.activate(ignoringOtherApps: true)
    signal(SIGTERM) { _ in exit(0) }
    app.run()
    exit(0)
}
