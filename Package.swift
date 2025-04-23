// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "BitNetSwift",
    products: [
        .library(
            name: "BitNetSwift",
            targets: ["BitNetSwift"]),
    ],
    targets: [
        .target(
            name: "BitNet",
            path: "Sources/BitNet",
            publicHeadersPath: "include"
        ),
        .target(
            name: "BitNetSwift",
            dependencies: ["BitNet"]
        ),
        .testTarget(
            name: "BitNetSwiftTests",
            dependencies: ["BitNetSwift"]
        ),
    ]
)
