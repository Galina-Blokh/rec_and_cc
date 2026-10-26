// `rcc-audiotap hotkey "<cmd>+<shift>+r"`: a system-wide keyboard shortcut.
//
// Uses Carbon's RegisterEventHotKey, which needs no Accessibility / Input Monitoring
// permission (unlike NSEvent global monitors or CGEventTaps): the window server delivers
// only the registered combination to us, never other keystrokes. It matches the physical
// key (virtual key code), so ⌘⇧R works the same with an English or a Russian layout.
//
// Output (one JSON object per line, flushed):
//   {"hotkey":"registered","combo":"<cmd>+<shift>+r"}   once, after a successful registration
//   {"hotkey":"pressed"}                                 per key press (no auto-repeat)
//   {"error":"taken"|"invalid"|"failed","message":...}  then exit 2
// Exits when stdin reaches EOF (the parent closed it or died), on SIGPIPE when stdout is
// closed, or on SIGTERM.

import AppKit
import Carbon.HIToolbox
import Foundation

struct HotkeyCombo {
    let keyCode: UInt32
    let modifiers: UInt32
}

enum HotkeyParseError: Error, CustomStringConvertible {
    case invalid(String)
    var description: String {
        switch self { case .invalid(let message): return message }
    }
}

private let modifierTokens: [String: Int] = [
    "cmd": cmdKey, "command": cmdKey, "super": cmdKey,
    "shift": shiftKey,
    "alt": optionKey, "option": optionKey, "opt": optionKey,
    "ctrl": controlKey, "control": controlKey,
]

private let keyCodes: [String: Int] = [
    "a": kVK_ANSI_A, "b": kVK_ANSI_B, "c": kVK_ANSI_C, "d": kVK_ANSI_D, "e": kVK_ANSI_E,
    "f": kVK_ANSI_F, "g": kVK_ANSI_G, "h": kVK_ANSI_H, "i": kVK_ANSI_I, "j": kVK_ANSI_J,
    "k": kVK_ANSI_K, "l": kVK_ANSI_L, "m": kVK_ANSI_M, "n": kVK_ANSI_N, "o": kVK_ANSI_O,
    "p": kVK_ANSI_P, "q": kVK_ANSI_Q, "r": kVK_ANSI_R, "s": kVK_ANSI_S, "t": kVK_ANSI_T,
    "u": kVK_ANSI_U, "v": kVK_ANSI_V, "w": kVK_ANSI_W, "x": kVK_ANSI_X, "y": kVK_ANSI_Y,
    "z": kVK_ANSI_Z,
    "0": kVK_ANSI_0, "1": kVK_ANSI_1, "2": kVK_ANSI_2, "3": kVK_ANSI_3, "4": kVK_ANSI_4,
    "5": kVK_ANSI_5, "6": kVK_ANSI_6, "7": kVK_ANSI_7, "8": kVK_ANSI_8, "9": kVK_ANSI_9,
    "f1": kVK_F1, "f2": kVK_F2, "f3": kVK_F3, "f4": kVK_F4, "f5": kVK_F5, "f6": kVK_F6,
    "f7": kVK_F7, "f8": kVK_F8, "f9": kVK_F9, "f10": kVK_F10, "f11": kVK_F11, "f12": kVK_F12,
    "space": kVK_Space,
]

/// Parses "<cmd>+<shift>+r" (pynput-style; case-insensitive, `<...>` optional around
/// modifiers and named keys). Exactly one non-modifier key is required. Function keys may be
/// used without modifiers; everything else needs at least one.
func parseHotkey(_ text: String) throws -> HotkeyCombo {
    var modifiers = 0
    var key: Int?
    let parts = text.lowercased().split(separator: "+", omittingEmptySubsequences: false)
    for raw in parts {
        var token = raw.trimmingCharacters(in: .whitespaces)
        if token.hasPrefix("<") && token.hasSuffix(">") && token.count >= 2 {
            token = String(token.dropFirst().dropLast())
        }
        if token.isEmpty { throw HotkeyParseError.invalid("empty part in \"\(text)\"") }
        if let mod = modifierTokens[token] {
            modifiers |= mod
        } else if let code = keyCodes[token] {
            if key != nil { throw HotkeyParseError.invalid("more than one key in \"\(text)\"") }
            key = code
        } else {
            throw HotkeyParseError.invalid("unknown key \"\(token)\" in \"\(text)\"")
        }
    }
    guard let key else { throw HotkeyParseError.invalid("no key in \"\(text)\"") }
    let isFunctionKey = keyCodes.contains { $0.value == key && $0.key.hasPrefix("f") && $0.key.count > 1 }
    if modifiers == 0 && !isFunctionKey {
        throw HotkeyParseError.invalid("\"\(text)\" needs a modifier (<cmd>, <ctrl>, <alt>, <shift>)")
    }
    return HotkeyCombo(keyCode: UInt32(key), modifiers: UInt32(modifiers))
}

private func emit(_ object: [String: Any]) {
    printJSON(object)  // flushes; SIGPIPE ends us if the reader is gone
}

private var hotKeyRef: EventHotKeyRef?

func runHotkey(_ text: String) -> Never {
    let combo: HotkeyCombo
    do {
        combo = try parseHotkey(text)
    } catch {
        emit(["error": "invalid", "message": "\(error)"])
        exit(2)
    }

    // A window-server connection and a running event loop deliver the hot key events.
    // .accessory (= LSUIElement): no Dock icon and no menu bar; we never open a window.
    let app = NSApplication.shared
    app.setActivationPolicy(.accessory)

    var eventType = EventTypeSpec(eventClass: OSType(kEventClassKeyboard),
                                  eventKind: UInt32(kEventHotKeyPressed))
    let handlerStatus = InstallEventHandler(GetApplicationEventTarget(), { _, _, _ in
        emit(["hotkey": "pressed"])
        return noErr
    }, 1, &eventType, nil, nil)
    guard handlerStatus == noErr else {
        emit(["error": "failed", "message": "InstallEventHandler failed (OSStatus \(handlerStatus))"])
        exit(2)
    }

    let id = EventHotKeyID(signature: OSType(0x5243_4348), id: 1)  // 'RCCH'
    let status = RegisterEventHotKey(combo.keyCode, combo.modifiers, id, GetApplicationEventTarget(),
                                     OptionBits(kEventHotKeyExclusive), &hotKeyRef)
    switch status {
    case noErr:
        break
    case OSStatus(eventHotKeyExistsErr):
        emit(["error": "taken", "message": "\(text) is already registered by another app"])
        exit(2)
    case OSStatus(eventHotKeyInvalidErr):
        emit(["error": "invalid", "message": "\(text) cannot be used as a global shortcut"])
        exit(2)
    default:
        emit(["error": "failed", "message": "RegisterEventHotKey failed (OSStatus \(status))"])
        exit(2)
    }
    emit(["hotkey": "registered", "combo": text])

    // Tie our lifetime to the parent: it holds our stdin open and EOF means it is gone.
    Thread.detachNewThread {
        _ = FileHandle.standardInput.readDataToEndOfFile()
        exit(0)
    }
    signal(SIGTERM) { _ in exit(0) }
    app.run()
    exit(0)
}
