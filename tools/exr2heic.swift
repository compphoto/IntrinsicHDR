// exr2heic.swift
//
// Convert a scene-linear EXR into an Apple Adaptive-HDR (ISO gain-map) HEIC
// with a Display P3 base color space.
//
// Verified API facts (compile-tested against MacOSX.sdk, ImageIO.framework):
//   - Image I/O decodes `.exr` natively via CGImageSourceCreateWithURL into an
//     extended-range linear sRGB (Rec.709) half-float CGImage; values > 1.0 are
//     preserved. We keep the image in its native extended/linear space and do
//     NOT clamp to SDR.
//   - HDR write path (CGImageDestination.h):
//       * kCGImageDestinationLossyCompressionQuality            (Double)
//       * kCGImageDestinationEncodeRequest                      -> kCGImageDestinationEncodeToISOGainmap
//       * kCGImageDestinationEncodeRequestOptions               (CFDictionary) containing
//             kCGImageDestinationEncodeBaseColorSpace           -> CGColorSpace(name: .displayP3)
//
// Usage:
//   exr2heic <input.exr> <output.heic> [--quality <0..1, default 0.9>]

import Foundation
import ImageIO
import CoreGraphics
import UniformTypeIdentifiers

// MARK: - Errors

/// Domain errors surfaced to the CLI; each maps to a clear stderr message and a
/// nonzero exit code.
enum Exr2HeicError: Error, CustomStringConvertible {
    case inputNotFound(String)
    case decodeFailed(String)
    case destinationCreateFailed(String)
    case finalizeFailed(String)

    var description: String {
        switch self {
        case .inputNotFound(let p):          return "input file not found: \(p)"
        case .decodeFailed(let p):           return "failed to decode EXR: \(p)"
        case .destinationCreateFailed(let p):return "failed to create HEIC destination: \(p)"
        case .finalizeFailed(let p):         return "failed to finalize HEIC: \(p)"
        }
    }
}

// MARK: - Load

/// Load the EXR as a native extended-range linear half-float CGImage.
///
/// Image I/O decodes `.exr` directly; we deliberately pass no decode options so
/// the extended/linear color space and values > 1.0 are preserved. Returns the
/// image at index 0. Throws rather than force-unwrapping on user-controlled paths.
func loadHDR(from inputURL: URL) throws -> CGImage {
    guard FileManager.default.fileExists(atPath: inputURL.path) else {
        throw Exr2HeicError.inputNotFound(inputURL.path)
    }
    // CGImageSourceCreateWithURL returns nil for unreadable / unsupported files.
    guard let source = CGImageSourceCreateWithURL(inputURL as CFURL, nil) else {
        throw Exr2HeicError.decodeFailed(inputURL.path)
    }
    guard CGImageSourceGetCount(source) > 0,
          let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
        throw Exr2HeicError.decodeFailed(inputURL.path)
    }
    return image
}

// MARK: - Encode

/// Encode an extended-range/linear CGImage to an Adaptive-HDR (ISO gain-map)
/// HEIC with a Display P3 base color space.
func encode(_ image: CGImage, to outputURL: URL, quality: Double) throws {
    // Ensure the parent directory exists.
    let parent = outputURL.deletingLastPathComponent()
    try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)

    guard let destination = CGImageDestinationCreateWithURL(
        outputURL as CFURL,
        UTType.heic.identifier as CFString,
        1,
        nil
    ) else {
        throw Exr2HeicError.destinationCreateFailed(outputURL.path)
    }

    // The named-space unwrap is acceptable per spec: displayP3 is a built-in
    // system color space that is always present.
    let baseP3 = CGColorSpace(name: CGColorSpace.displayP3)!

    // kCGImageDestinationEncodeRequestOptions sub-dictionary.
    // NOTE on 10-bit base: kCGImageDestinationEncodeBasePixelFormatRequest exists
    // in CGImageDestination.h, but the SDK headers expose NO value constants for
    // it (no RGB10 / pixel-format enums anywhere in ImageIO.framework/Headers),
    // so the correct value/type is unclear. Per spec we omit it and leave the
    // base at its default rather than guess.
    let encodeRequestOptions: [CFString: Any] = [
        kCGImageDestinationEncodeBaseColorSpace: baseP3
    ]

    let properties: [CFString: Any] = [
        kCGImageDestinationLossyCompressionQuality: quality,
        kCGImageDestinationEncodeRequest: kCGImageDestinationEncodeToISOGainmap as Any,
        kCGImageDestinationEncodeRequestOptions: encodeRequestOptions
    ]

    CGImageDestinationAddImage(destination, image, properties as CFDictionary)

    guard CGImageDestinationFinalize(destination) else {
        throw Exr2HeicError.finalizeFailed(outputURL.path)
    }
}

// MARK: - CLI

let usage = """
usage: exr2heic <input.exr> <output.heic> [--quality <0..1, default 0.9>]
  Converts a scene-linear EXR into an Apple Adaptive-HDR (ISO gain-map) HEIC
  with a Display P3 base color space.
"""

/// Parsed command-line arguments.
struct Arguments {
    let input: URL
    let output: URL
    let quality: Double
}

/// Parse argv (excluding the program name). Returns nil if invalid.
func parseArguments(_ args: [String]) -> Arguments? {
    var positional: [String] = []
    var quality: Double = 0.9
    var i = 0
    while i < args.count {
        let arg = args[i]
        if arg == "--quality" {
            guard i + 1 < args.count,
                  let q = Double(args[i + 1]),
                  q >= 0.0, q <= 1.0 else {
                return nil
            }
            quality = q
            i += 2
        } else if arg == "-h" || arg == "--help" {
            return nil
        } else {
            positional.append(arg)
            i += 1
        }
    }
    guard positional.count == 2 else { return nil }
    return Arguments(
        input: URL(fileURLWithPath: positional[0]),
        output: URL(fileURLWithPath: positional[1]),
        quality: quality
    )
}

func main() -> Int32 {
    let argv = Array(CommandLine.arguments.dropFirst())
    guard let arguments = parseArguments(argv) else {
        FileHandle.standardError.write(Data((usage + "\n").utf8))
        return 2
    }

    do {
        let image = try loadHDR(from: arguments.input)
        try encode(image, to: arguments.output, quality: arguments.quality)

        let size = (try? FileManager.default
            .attributesOfItem(atPath: arguments.output.path)[.size] as? Int) ?? nil
        let bytes = size ?? 0
        print("wrote \(arguments.output.path) (\(bytes) bytes)")
        return 0
    } catch {
        FileHandle.standardError.write(Data(("error: \(error)\n").utf8))
        return 1
    }
}

exit(main())
