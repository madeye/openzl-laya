// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "OpenZLLayaWorker",
    platforms: [.macOS(.v14)],
    products: [.executable(name: "openzl-laya-worker", targets: ["OpenZLLayaWorker"])],
    dependencies: [.package(url: "https://github.com/FluidInference/FluidUse.git", exact: "0.2.0")],
    targets: [
        .executableTarget(name: "OpenZLLayaWorker", dependencies: [.product(name: "FluidUse", package: "FluidUse")]),
        .testTarget(name: "OpenZLLayaWorkerTests", dependencies: ["OpenZLLayaWorker"]),
    ]
)
