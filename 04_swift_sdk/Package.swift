
// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "YOLOv11CoreMLSDK",
    platforms: [
        .iOS(.v15),
        .macOS(.v12)
    ],
    products: [
        .library(
            name: "YOLOv11CoreMLSDK",
            targets: ["YOLOv11CoreMLSDK"]),
    ],
    dependencies: [],
    targets: [
        .target(
            name: "YOLOv11CoreMLSDK",
            dependencies: [],
            resources: [
                .process("Resources/yolo11n.mlpackage")
            ]
        )
    ]
)
