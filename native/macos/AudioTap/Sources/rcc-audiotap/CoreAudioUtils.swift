// Thin, throwing wrappers around the Core Audio HAL property API.

import CoreAudio
import Foundation

/// A failed Core Audio call, with the operation that failed and a decoded OSStatus.
struct CoreAudioError: Error, CustomStringConvertible {
    let status: OSStatus
    let operation: String

    var description: String {
        "\(operation) failed: OSStatus \(status) \(fourCC(status))"
    }
}

/// Generic error for everything that is not an OSStatus.
struct ToolError: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
}

/// Core Audio errors are usually four-character codes ('who?', '!obj', ...). Render them
/// readably when they are, so error messages can be looked up in the AudioHardware headers.
func fourCC(_ status: OSStatus) -> String {
    let value = UInt32(bitPattern: status)
    let bytes = [24, 16, 8, 0].map { UInt8((value >> UInt32($0)) & 0xFF) }
    guard bytes.allSatisfy({ $0 >= 0x20 && $0 < 0x7F }) else { return "" }
    return "('\(String(decoding: bytes, as: UTF8.self))')"
}

@inline(__always)
func check(_ status: OSStatus, _ operation: @autoclosure () -> String) throws {
    guard status == noErr else { throw CoreAudioError(status: status, operation: operation()) }
}

func propertyAddress(
    _ selector: AudioObjectPropertySelector,
    scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal,
    element: AudioObjectPropertyElement = kAudioObjectPropertyElementMain
) -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress(mSelector: selector, mScope: scope, mElement: element)
}

/// Read a fixed-size (POD) property such as a UInt32, pid_t or AudioStreamBasicDescription.
func readValue<T>(_ object: AudioObjectID, _ selector: AudioObjectPropertySelector, as _: T.Type,
                  initial: T, name: String) throws -> T {
    var address = propertyAddress(selector)
    var size = UInt32(MemoryLayout<T>.size)
    var value = initial
    try withUnsafeMutablePointer(to: &value) { ptr in
        try check(AudioObjectGetPropertyData(object, &address, 0, nil, &size, ptr),
                  "Reading \(name) of audio object \(object)")
    }
    return value
}

func readUInt32(_ object: AudioObjectID, _ selector: AudioObjectPropertySelector, name: String) throws -> UInt32 {
    try readValue(object, selector, as: UInt32.self, initial: 0, name: name)
}

/// Read a CFString property. Core Audio returns these with "copy" semantics (+1 retained),
/// so we take ownership of the reference.
func readString(_ object: AudioObjectID, _ selector: AudioObjectPropertySelector, name: String) throws -> String? {
    var address = propertyAddress(selector)
    var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
    var ref: Unmanaged<CFString>? = nil
    try withUnsafeMutablePointer(to: &ref) { ptr in
        try check(AudioObjectGetPropertyData(object, &address, 0, nil, &size, ptr),
                  "Reading \(name) of audio object \(object)")
    }
    return ref?.takeRetainedValue() as String?
}

/// Read a variable-length array of AudioObjectIDs (device lists, process lists, ...).
func readObjectList(_ object: AudioObjectID, _ selector: AudioObjectPropertySelector, name: String) throws -> [AudioObjectID] {
    var address = propertyAddress(selector)
    var size: UInt32 = 0
    try check(AudioObjectGetPropertyDataSize(object, &address, 0, nil, &size), "Sizing \(name)")
    let count = Int(size) / MemoryLayout<AudioObjectID>.size
    guard count > 0 else { return [] }
    var list = [AudioObjectID](repeating: kAudioObjectUnknown, count: count)
    try check(AudioObjectGetPropertyData(object, &address, 0, nil, &size, &list), "Reading \(name)")
    // The list can shrink between the two calls.
    return Array(list.prefix(Int(size) / MemoryLayout<AudioObjectID>.size))
}

/// Map a Unix PID to its Core Audio process object. Returns nil when the process has no
/// process object (it has never touched Core Audio), which is not an error.
func processObject(forPID pid: pid_t) throws -> AudioObjectID? {
    var address = propertyAddress(kAudioHardwarePropertyTranslatePIDToProcessObject)
    var qualifier = pid
    var object = AudioObjectID(kAudioObjectUnknown)
    var size = UInt32(MemoryLayout<AudioObjectID>.size)
    try check(AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &address,
                                         UInt32(MemoryLayout<pid_t>.size), &qualifier, &size, &object),
              "Translating PID \(pid) to a Core Audio process object")
    return object == kAudioObjectUnknown ? nil : object
}

/// Process taps and kAudioHardwarePropertyProcessObjectList exist since macOS 14.4.
func requireProcessTapSupport() throws {
    let v = ProcessInfo.processInfo.operatingSystemVersion
    guard v.majorVersion > 14 || (v.majorVersion == 14 && v.minorVersion >= 4) else {
        throw ToolError("Core Audio process taps need macOS 14.4 or newer (this is \(v.majorVersion).\(v.minorVersion).\(v.patchVersion))")
    }
}

func logError(_ message: String) {
    FileHandle.standardError.write(Data("rcc-audiotap: \(message)\n".utf8))
}
