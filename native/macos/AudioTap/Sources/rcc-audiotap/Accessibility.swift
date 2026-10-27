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

private final class Walker {
    var nodes = 0
    var names: [String] = []
    let deadline = Date().addingTimeInterval(3)

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

    /// The first text of each row of `list`. True if any was found.
    private func collect(_ list: AXUIElement) -> Bool {
        var found = false
        func rows(_ element: AXUIElement, _ depth: Int) {
            if exhausted || depth > 8 { return }
            nodes += 1
            for row in children(element) {
                let role = text(row, kAXRoleAttribute)
                if role == "AXRow" || role == "AXListItem" {
                    if let name = firstText(row, 0) {
                        names.append(name)
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

    private func firstText(_ element: AXUIElement, _ depth: Int) -> String? {
        if exhausted || depth > 6 { return nil }
        nodes += 1
        if depth > 0, textRoles.contains(text(element, kAXRoleAttribute)) {
            for name in [kAXValueAttribute, kAXTitleAttribute, kAXDescriptionAttribute] {
                let value = text(element, name)
                if !value.isEmpty { return value }
            }
        } else if depth == 0 {
            let value = text(element, kAXTitleAttribute)
            if !value.isEmpty { return value }
        }
        for child in children(element) {
            if let value = firstText(child, depth + 1) { return value }
        }
        return nil
    }
}

/// Prints {"trusted": bool, "names": [...]} for the visible participant list of the running
/// apps whose pid is in `pids` or whose bundle id / name contains one of `matches`.
func runParticipants(pids: [pid_t], matches: [String]) -> Never {
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
    var names: [String] = []
    for app in apps {
        let walker = Walker()
        let root = AXUIElementCreateApplication(app.processIdentifier)
        AXUIElementSetMessagingTimeout(root, 1.0)
        walker.find(root, depth: 0)
        names += walker.names
    }
    printJSON(["trusted": true, "names": names])
    exit(0)
}
