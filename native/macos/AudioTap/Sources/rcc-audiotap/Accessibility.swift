// `rcc-audiotap ax-status` and `rcc-audiotap participants`: read-only Accessibility access, used
// only for the opt-in "names from the call app" feature.
//
// Trust is only ever checked silently (AXIsProcessTrusted). This file never calls
// AXIsProcessTrustedWithOptions, so the helper can't make macOS show the Accessibility prompt;
// the user is sent to System Settings by an explicit menu action instead.
// Nothing is clicked, focused or opened: only attributes of already visible elements are read.

import AppKit
import ApplicationServices
import Foundation

func runAXStatus() -> Never {
    printJSON(["trusted": AXIsProcessTrusted()])
    exit(0)
}

private func attribute(_ element: AXUIElement, _ name: String) -> CFTypeRef? {
    var value: CFTypeRef?
    return AXUIElementCopyAttributeValue(element, name as CFString, &value) == .success ? value : nil
}

private func text(_ element: AXUIElement, _ name: String) -> String {
    (attribute(element, name) as? String)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
}

private func children(_ element: AXUIElement) -> [AXUIElement] {
    (attribute(element, kAXChildrenAttribute) as? [AXUIElement]) ?? []
}

private let listWords = ["participant", "people", "attendee", "участник", "משתתפ"]
private let listRoles: Set<String> = ["AXList", "AXTable", "AXOutline", "AXGroup", "AXScrollArea"]
private let textRoles: Set<String> = ["AXStaticText", "AXCell", "AXRow", "AXListItem", "AXGroup"]
private let maxNodes = 4000
private let maxDepth = 30
private let perAppSeconds = 2.0

// Browsers show many tabs and windows: only a window whose title says it is a call page is
// read, never a LinkedIn/Contacts/GitHub "People" list in another tab. (A window's title is
// the title of its selected tab.)
private let browserIDs = [
    "chrome", "safari", "firefox", "edgemac", "brave", "thebrowser", "opera", "vivaldi",
]
private let callPageWords = [
    "google meet", "meet.google.com", "meet - ", "meet \u{2013} ", "meet \u{2014} ",
    "microsoft teams", "teams.microsoft", "zoom",
]

private final class Walker {
    var nodes = 0
    var names: [String] = []
    let deadline = Date().addingTimeInterval(perAppSeconds)  // each app has its own budget

    var exhausted: Bool { nodes >= maxNodes || Date() > deadline }

    /// Looks for a list-like element labelled "Participants"/"People" and reads its rows.
    func find(_ element: AXUIElement, depth: Int) {
        if exhausted || depth > maxDepth { return }
        nodes += 1
        let role = text(element, kAXRoleAttribute)
        let label = (text(element, kAXTitleAttribute) + " " + text(element, kAXDescriptionAttribute))
            .lowercased()
        if listRoles.contains(role), listWords.contains(where: { label.contains($0) }), collect(element) {
            return
        }
        for child in children(element) { find(child, depth: depth + 1) }
    }

    /// One entry per row of `list`: all the row's texts, one per line. True if any was found.
    private func collect(_ list: AXUIElement) -> Bool {
        var found = false
        func rows(_ element: AXUIElement, _ depth: Int) {
            if exhausted || depth > 8 { return }
            nodes += 1
            for row in children(element) {
                let role = text(row, kAXRoleAttribute)
                if role == "AXRow" || role == "AXListItem" {
                    var texts: [String] = []
                    allTexts(row, 0, &texts)
                    if !texts.isEmpty {
                        // every text of the row, so the caller can spot the user's own "(You)"
                        names.append(texts.joined(separator: "\n"))
                        found = true
                    }
                } else {
                    rows(row, depth + 1)
                }
            }
        }
        rows(list, 0)
        return found
    }

    private func allTexts(_ element: AXUIElement, _ depth: Int, _ out: inout [String]) {
        if exhausted || depth > 6 || out.count >= 8 { return }
        nodes += 1
        if textRoles.contains(text(element, kAXRoleAttribute)) {
            for name in [kAXValueAttribute, kAXTitleAttribute, kAXDescriptionAttribute] {
                let value = text(element, name)
                if !value.isEmpty {
                    if !out.contains(value) { out.append(value) }
                    break
                }
            }
        }
        for child in children(element) { allTexts(child, depth + 1, &out) }
    }
}

/// Defense in depth, not a security boundary: `participants` runs only when started by the
/// RecAndCC app's Python, which passes its own pid. (macOS attributes Accessibility to the
/// responsible app that spawned the helper, so another process's run is not covered by
/// RecAndCC's grant anyway; this just refuses the obvious misuse.)
private func startedByRecAndCC(_ expected: pid_t?) -> Bool {
    guard let expected, getppid() == expected else { return false }
    var buffer = [CChar](repeating: 0, count: 4096)
    guard proc_pidpath(expected, &buffer, UInt32(buffer.count)) > 0 else { return false }
    let path = String(cString: buffer).lowercased()
    return path.contains("python") || path.contains("recandcc")
}

/// Prints {"trusted": bool, "names": [...]} for the visible participant list of the running
/// apps whose pid is in `pids` or whose bundle id / name contains one of `matches`.
func runParticipants(pids: [pid_t], matches: [String], parentPID: pid_t?) -> Never {
    guard startedByRecAndCC(parentPID) else {
        logError("participants runs only when started by RecAndCC")
        printJSON(["trusted": false, "names": [String](), "refused": true])
        exit(3)
    }
    guard AXIsProcessTrusted() else {
        printJSON(["trusted": false, "names": [String]()])
        exit(0)
    }
    let wanted = matches.map { $0.lowercased() }
    let apps = NSWorkspace.shared.runningApplications.filter { app in
        // Regular apps only: audio helper processes have no windows of their own.
        guard app.activationPolicy == .regular else { return false }
        let id = (app.bundleIdentifier ?? "").lowercased()
        let name = (app.localizedName ?? "").lowercased()
        return pids.contains(app.processIdentifier)
            || wanted.contains { id.contains($0) || name.contains($0) }
    }
    // A slow app can't hang the whole read: a short timeout for every AX call of this process
    // and a separate time budget per app.
    AXUIElementSetMessagingTimeout(AXUIElementCreateSystemWide(), 0.5)
    var names: [String] = []
    for app in apps {
        let walker = Walker()
        let root = AXUIElementCreateApplication(app.processIdentifier)
        // The system-wide timeout is not inherited by application elements: set it on each.
        AXUIElementSetMessagingTimeout(root, 0.5)
        let id = (app.bundleIdentifier ?? "").lowercased()
        if browserIDs.contains(where: { id.contains($0) }) {
            let windows = (attribute(root, kAXWindowsAttribute) as? [AXUIElement]) ?? []
            for window in windows {
                let title = text(window, kAXTitleAttribute).lowercased()
                if callPageWords.contains(where: { title.contains($0) }) {
                    AXUIElementSetMessagingTimeout(window, 0.5)
                    walker.find(window, depth: 0)
                }
            }
        } else {
            walker.find(root, depth: 0)
        }
        names += walker.names
    }
    printJSON(["trusted": true, "names": names])
    exit(0)
}
