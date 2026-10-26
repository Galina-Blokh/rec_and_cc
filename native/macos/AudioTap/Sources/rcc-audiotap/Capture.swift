// System audio capture through a Core Audio process tap.
//
// How it fits together:
//   1. A CATapDescription describes *what* to tap: here a global stereo mixdown of every
//      process's output, minus an exclusion list (ourselves + whatever the caller passes).
//   2. AudioHardwareCreateProcessTap turns it into a tap audio object.
//   3. A tap cannot be read directly: it has to be placed inside an aggregate device
//      (kAudioAggregateDeviceTapListKey). The aggregate shows up with the tap as an *input*
//      stream. We make it private (invisible to other apps) and clock it from the default
//      output device.
//   4. An IOProc on the aggregate device receives the tapped audio in its input buffer list.
//      We copy it off the real-time thread and convert it to 16 kHz mono float32.

import AVFoundation
import CoreAudio
import Foundation

/// Output format handed to consumers: what Whisper wants.
let outputFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16_000, channels: 1, interleaved: false)!

final class SystemAudioCapture {
    /// Called on a private serial queue with each converted chunk (16 kHz mono float32).
    typealias Sink = (UnsafeBufferPointer<Float>) -> Void

    private let excludePIDs: [pid_t]
    private let sink: Sink
    private let queue = DispatchQueue(label: "rcc-audiotap.convert", qos: .userInitiated)

    private var tapID = AudioObjectID(kAudioObjectUnknown)
    private var aggregateID = AudioObjectID(kAudioObjectUnknown)
    private var ioProcID: AudioDeviceIOProcID?
    private var tapFormat: AVAudioFormat!
    private var converter: AVAudioConverter!
    private var loggedLayout = false

    init(excludePIDs: [pid_t], sink: @escaping Sink) {
        self.excludePIDs = excludePIDs
        self.sink = sink
    }

    deinit { stop() }

    func start() throws {
        try requireProcessTapSupport()
        do {
            try createTap()
            try createAggregateDevice()
            try setUpConverter()
            try startIO()
        } catch {
            stop()
            throw error
        }
    }

    /// Tear everything down in reverse order. Safe to call more than once.
    func stop() {
        if let ioProcID {
            AudioDeviceStop(aggregateID, ioProcID)
            let status = AudioDeviceDestroyIOProcID(aggregateID, ioProcID)
            if status != noErr { logError("AudioDeviceDestroyIOProcID: OSStatus \(status) \(fourCC(status))") }
            self.ioProcID = nil
        }
        if aggregateID != kAudioObjectUnknown {
            let status = AudioHardwareDestroyAggregateDevice(aggregateID)
            if status != noErr { logError("AudioHardwareDestroyAggregateDevice: OSStatus \(status) \(fourCC(status))") }
            aggregateID = AudioObjectID(kAudioObjectUnknown)
        }
        if tapID != kAudioObjectUnknown {
            let status = AudioHardwareDestroyProcessTap(tapID)
            if status != noErr { logError("AudioHardwareDestroyProcessTap: OSStatus \(status) \(fourCC(status))") }
            tapID = AudioObjectID(kAudioObjectUnknown)
        }
    }

    // MARK: - Setup steps

    private func createTap() throws {
        // Exclusions are expressed as Core Audio *process objects*, not PIDs. A PID that has
        // never used Core Audio has no process object - and produces no audio - so skip it.
        var excluded: [AudioObjectID] = []
        for pid in excludePIDs {
            if let object = try processObject(forPID: pid) {
                excluded.append(object)
            }
        }

        let description = CATapDescription(stereoGlobalTapButExcludeProcesses: excluded)
        description.name = "rcc-audiotap"
        description.uuid = UUID()
        // Private: only this process can see the tap. Unmuted: the user keeps hearing the audio.
        description.isPrivate = true
        description.muteBehavior = .unmuted

        var tap = AudioObjectID(kAudioObjectUnknown)
        try check(AudioHardwareCreateProcessTap(description, &tap), "AudioHardwareCreateProcessTap")
        tapID = tap

        var asbd = try readValue(tapID, kAudioTapPropertyFormat, as: AudioStreamBasicDescription.self,
                                 initial: AudioStreamBasicDescription(), name: "tap format")
        guard let format = AVAudioFormat(streamDescription: &asbd) else {
            throw ToolError("Unsupported tap format: \(asbd)")
        }
        tapFormat = format
        logError("tap format: \(format) (excluding \(excluded.count) process object(s))")
    }

    private func createAggregateDevice() throws {
        let outputDevice = try readValue(AudioObjectID(kAudioObjectSystemObject),
                                         kAudioHardwarePropertyDefaultOutputDevice,
                                         as: AudioDeviceID.self, initial: kAudioObjectUnknown,
                                         name: "default output device")
        guard outputDevice != kAudioObjectUnknown else { throw ToolError("No default output device") }
        guard let outputUID = try readString(outputDevice, kAudioDevicePropertyDeviceUID, name: "device UID") else {
            throw ToolError("Default output device has no UID")
        }
        let tapUID = try readString(tapID, kAudioTapPropertyUID, name: "tap UID") ?? ""

        let description: [String: Any] = [
            kAudioAggregateDeviceNameKey: "rcc-audiotap",
            kAudioAggregateDeviceUIDKey: "com.gblokh.recandcc.audiotap.\(UUID().uuidString)",
            // The main sub-device provides the clock the IOProc runs on.
            kAudioAggregateDeviceMainSubDeviceKey: outputUID,
            kAudioAggregateDeviceIsPrivateKey: true,
            kAudioAggregateDeviceIsStackedKey: false,
            // Start the tap together with the aggregate device's IO.
            kAudioAggregateDeviceTapAutoStartKey: true,
            kAudioAggregateDeviceSubDeviceListKey: [
                [kAudioSubDeviceUIDKey: outputUID],
            ],
            kAudioAggregateDeviceTapListKey: [
                [
                    kAudioSubTapUIDKey: tapUID,
                    kAudioSubTapDriftCompensationKey: true,
                ],
            ],
        ]
        var aggregate = AudioObjectID(kAudioObjectUnknown)
        try check(AudioHardwareCreateAggregateDevice(description as CFDictionary, &aggregate),
                  "AudioHardwareCreateAggregateDevice (main sub-device \(outputUID))")
        aggregateID = aggregate
    }

    private func setUpConverter() throws {
        guard let converter = AVAudioConverter(from: tapFormat, to: outputFormat) else {
            throw ToolError("Cannot convert from \(tapFormat!) to \(outputFormat)")
        }
        // Mix all channels into mono instead of just taking the first one.
        converter.downmix = true
        self.converter = converter
    }

    private func startIO() throws {
        var procID: AudioDeviceIOProcID?
        try check(AudioDeviceCreateIOProcIDWithBlock(&procID, aggregateID, nil) { [weak self] _, inputData, _, _, _ in
            self?.handleInput(inputData)
        }, "AudioDeviceCreateIOProcIDWithBlock")
        ioProcID = procID
        try check(AudioDeviceStart(aggregateID, procID), "AudioDeviceStart on the tap aggregate device")
    }

    // MARK: - Audio path

    /// Real-time thread: copy the tap's samples into an owned buffer and get off this thread.
    private func handleInput(_ inputData: UnsafePointer<AudioBufferList>) {
        let buffers = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: inputData))
        // The tap's stream(s) come after any input streams of the sub-device (e.g. a headset
        // mic on the output device), so take the trailing buffers that match the tap format.
        let needed = tapFormat.isInterleaved ? 1 : Int(tapFormat.channelCount)
        guard buffers.count >= needed else {
            if !loggedLayout {
                loggedLayout = true
                let count = buffers.count
                queue.async { logError("IO buffer list has \(count) buffer(s), expected \(needed); no tap input") }
            }
            return
        }
        let tapBuffers = Array(buffers.suffix(needed))
        let bytesPerFrame = Int(tapFormat.streamDescription.pointee.mBytesPerFrame)
        guard bytesPerFrame > 0 else { return }
        let frames = Int(tapBuffers[0].mDataByteSize) / bytesPerFrame
        guard frames > 0,
              let copy = AVAudioPCMBuffer(pcmFormat: tapFormat, frameCapacity: AVAudioFrameCount(frames)) else { return }
        copy.frameLength = AVAudioFrameCount(frames)
        let destination = UnsafeMutableAudioBufferListPointer(copy.mutableAudioBufferList)
        for (src, var dst) in zip(tapBuffers, destination) {
            if let s = src.mData, let d = dst.mData {
                memcpy(d, s, Int(min(src.mDataByteSize, dst.mDataByteSize)))
            }
            dst.mDataByteSize = src.mDataByteSize
        }
        if !loggedLayout {
            loggedLayout = true
            let layout = buffers.count
            queue.async { logError("IO buffer list: \(layout) buffer(s), using last \(needed); \(frames) frames/cycle") }
        }
        queue.async { [weak self] in self?.convert(copy) }
    }

    /// Conversion queue: resample/downmix to 16 kHz mono and hand to the sink.
    private func convert(_ input: AVAudioPCMBuffer) {
        let ratio = outputFormat.sampleRate / tapFormat.sampleRate
        let capacity = AVAudioFrameCount((Double(input.frameLength) * ratio).rounded(.up)) + 64
        guard let output = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: capacity) else { return }
        var consumed = false
        var error: NSError?
        // The converter keeps resampler state between calls, so feed exactly one buffer and
        // then say "no data *now*" (not end-of-stream, which would flush and reset it).
        let status = converter.convert(to: output, error: &error) { _, inputStatus in
            if consumed {
                inputStatus.pointee = .noDataNow
                return nil
            }
            consumed = true
            inputStatus.pointee = .haveData
            return input
        }
        if status == .error {
            logError("AVAudioConverter: \(error?.localizedDescription ?? "unknown error")")
            return
        }
        guard output.frameLength > 0, let samples = output.floatChannelData?[0] else { return }
        sink(UnsafeBufferPointer(start: samples, count: Int(output.frameLength)))
    }
}

// MARK: - `rcc-audiotap capture`

func runCapture(excludePIDs: [pid_t]) throws -> Never {
    // Writes to a closed pipe must return EPIPE instead of killing us, so we can clean up.
    signal(SIGPIPE, SIG_IGN)

    var capture: SystemAudioCapture?
    let shutdown: (Int32) -> Never = { code in
        capture?.stop()
        exit(code)
    }

    // Every write goes through this serial queue so real audio and silence filler never interleave.
    let writer = DispatchQueue(label: "rcc-audiotap.writer")
    let sampleRate = 16_000.0
    let startedAt = DispatchTime.now().uptimeNanoseconds
    var samplesWritten = 0
    var lastRealAudio = startedAt

    let emit: (Data) -> Void = { data in
        // native float32 = little-endian on Apple silicon/x86
        if !writeAll(STDOUT_FILENO, data) {
            DispatchQueue.main.async {
                logError("stdout closed, stopping")
                shutdown(0)
            }
        }
    }

    capture = SystemAudioCapture(excludePIDs: [getpid()] + excludePIDs) { samples in
        let data = Data(buffer: samples)
        writer.sync {
            lastRealAudio = DispatchTime.now().uptimeNanoseconds
            samplesWritten += samples.count
            emit(data)
        }
    }

    // A process tap delivers no IO cycles at all while nothing is playing. Fill those gaps with
    // zeros, paced by the wall clock, so the stream stays continuous: the "others" track then
    // stays aligned with the mic track, and the reader can tell "silence" from "stalled".
    let filler = DispatchSource.makeTimerSource(queue: writer)
    filler.schedule(deadline: .now() + 0.1, repeating: 0.1)
    filler.setEventHandler {
        let now = DispatchTime.now().uptimeNanoseconds
        guard now - lastRealAudio > 200_000_000 else { return }  // real audio is flowing
        let expected = Int(Double(now - startedAt) / 1e9 * sampleRate)
        let missing = expected - samplesWritten
        guard missing > 0 else { return }
        samplesWritten += missing
        emit(Data(count: missing * MemoryLayout<Float32>.size))
    }
    filler.resume()

    // Handle SIGINT/SIGTERM on the main queue (not in a signal handler) so teardown can call
    // into Core Audio safely.
    var sources: [DispatchSourceSignal] = []
    for sig in [SIGINT, SIGTERM] {
        signal(sig, SIG_IGN)
        let source = DispatchSource.makeSignalSource(signal: sig, queue: .main)
        source.setEventHandler {
            logError("signal \(sig), stopping")
            shutdown(0)
        }
        source.resume()
        sources.append(source)
    }

    try capture!.start()
    logError("capturing system audio as float32le mono 16000 Hz on stdout")
    withExtendedLifetime((sources, filler)) { dispatchMain() }
}
