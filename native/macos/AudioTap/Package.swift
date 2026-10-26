// swift-tools-version: 6.0
import PackageDescription
import Foundation

// Absolute path to the Info.plist that gets embedded into the binary's __TEXT,__info_plist
// section. A command-line tool has no bundle, so this is how TCC finds the usage strings
// (NSAudioCaptureUsageDescription / NSMicrophoneUsageDescription) and our bundle identifier.
let packageDir = URL(fileURLWithPath: #filePath).deletingLastPathComponent().path
let infoPlist = "\(packageDir)/Support/Info.plist"

let package = Package(
    name: "AudioTap",
    // Core Audio process taps and process objects (kAudioProcessPropertyIsRunning*) need 14.4+.
    platforms: [.macOS("14.4")],
    products: [
        .executable(name: "rcc-audiotap", targets: ["rcc-audiotap"]),
        // Contents/MacOS/RecAndCC of the app bundle (see ../build_app.sh).
        .executable(name: "RecAndCC", targets: ["RecAndCC"]),
    ],
    targets: [
        .executableTarget(
            name: "rcc-audiotap",
            path: "Sources/rcc-audiotap",
            swiftSettings: [.swiftLanguageMode(.v5)],
            linkerSettings: [
                .linkedFramework("CoreAudio"),
                .linkedFramework("AVFoundation"),
                .linkedFramework("AppKit"),
                .unsafeFlags([
                    "-Xlinker", "-sectcreate",
                    "-Xlinker", "__TEXT",
                    "-Xlinker", "__info_plist",
                    "-Xlinker", infoPlist,
                ]),
            ]
        ),
        .executableTarget(
            name: "RecAndCC",
            path: "Sources/RecAndCC",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
