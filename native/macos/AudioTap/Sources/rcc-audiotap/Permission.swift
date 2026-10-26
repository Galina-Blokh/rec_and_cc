// `rcc-audiotap permission`: best-effort check that system audio capture delivers real audio.
//
// There is no public API to query the "System Audio Recording" TCC permission. Without it a
// process tap still works but delivers pure digital silence. So we tap all audio (including
// our own), optionally play a short system sound, and look at whether anything non-zero
// arrived.

import AppKit
import Foundation

func runPermissionCheck(seconds: Double, playSound: Bool) throws -> Never {
    let lock = NSLock()
    var peak: Float = 0
    var samples = 0

    let capture = SystemAudioCapture(excludePIDs: []) { chunk in
        var localPeak: Float = 0
        for value in chunk { localPeak = max(localPeak, abs(value)) }
        lock.lock()
        peak = max(peak, localPeak)
        samples += chunk.count
        lock.unlock()
    }
    do {
        try capture.start()
    } catch {
        print("error: system audio capture could not start: \(error)")
        exit(1)
    }

    var sound: NSSound?
    if playSound {
        sound = NSSound(contentsOfFile: "/System/Library/Sounds/Tink.aiff", byReference: true)
        sound?.volume = 0.3
        sound?.play()
    }
    // NSSound needs a running run loop.
    RunLoop.main.run(until: Date().addingTimeInterval(seconds))
    sound?.stop()
    capture.stop()

    lock.lock()
    let (finalPeak, finalSamples) = (peak, samples)
    lock.unlock()

    if finalSamples == 0 {
        print("error: no audio frames received from the tap")
        exit(1)
    }
    if finalPeak > 0 {
        print("ok: system audio capture works (peak \(String(format: "%.4f", finalPeak)), \(finalSamples) samples)")
        exit(0)
    }
    print("""
    silent: received \(finalSamples) samples, all zero. Most likely the "System Audio Recording" \
    permission is missing for the app that launched rcc-audiotap (System Settings > Privacy & \
    Security > Screen & System Audio Recording)\(playSound ? "" : ", or nothing was playing").
    """)
    exit(1)
}
