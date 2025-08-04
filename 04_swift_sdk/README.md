# 第四步：Swift SDK 开发

这是完整项目流程的第四步，主要目标是将 CoreML 模型封装成 Swift Package，为 iOS/macOS 应用提供原生的目标检测功能。

## 🎯 目标

- 创建 Swift Package Manager 兼容的 SDK
- 提供原生 iOS/macOS API 接口
- 支持实时相机检测和图片检测
- 包含完整的错误处理和性能优化
- 提供 SwiftUI 视图组件

## 📋 环境要求

- Xcode 14.0+
- iOS 15.0+ / macOS 12.0+
- Swift 5.5+
- 已完成前三步的模型准备

## 🚀 快速开始

### 方法一：Swift Package Manager（推荐）

在 Xcode 中：
1. File → Add Package Dependencies
2. 输入仓库 URL
3. 选择版本并添加到项目

### 方法二：本地开发

```bash
cd 04_swift_sdk

# 在 Xcode 中打开 Package.swift
open Package.swift

# 或者运行测试
swift test
```

### 基本使用

```swift
import YOLOv11CoreMLSDK
import UIKit

// 在 ViewController 中
class ViewController: UIViewController {
    private var detector: YOLOv11Predictor!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // 初始化检测器
        do {
            detector = try YOLOv11Predictor()
        } catch {
            print("初始化失败: \(error)")
        }
    }
    
    func detectObjects(in image: UIImage) async {
        do {
            let detections = try await detector.predict(image: image)
            
            for detection in detections {
                print("\(detection.label): \(detection.confidence)")
            }
        } catch {
            print("检测失败: \(error)")
        }
    }
}
```

## 📁 文件结构

```
04_swift_sdk/
├── Package.swift                    # Swift Package 配置
├── Sources/
│   └── YOLOv11CoreMLSDK/
│       ├── YOLOv11CoreMLSDK.swift   # 主 SDK 入口
│       ├── YOLOv11Predictor.swift   # 核心预测器
│       ├── ObjectDetector.swift     # 检测逻辑
│       ├── DetectionView.swift      # SwiftUI 视图组件
│       └── Resources/
│           └── yolo11n.mlpackage    # CoreML 模型
├── Tests/
│   └── YOLOv11CoreMLSDKTests/
│       └── YOLOv11CoreMLSDKTests.swift
└── README.md
```

## 🔧 API 文档

### YOLOv11Predictor 类

主要的预测器类，提供目标检测功能。

#### 初始化

```swift
// 使用默认模型
let predictor = try YOLOv11Predictor()

// 使用自定义配置
let predictor = try YOLOv11Predictor(
    modelName: "yolo11n",
    confidenceThreshold: 0.5,
    iouThreshold: 0.45
)
```

#### 主要方法

##### `predict(image: UIImage) async throws -> [Detection]`

对单张图片进行目标检测。

**参数:**
- `image`: UIImage 对象

**返回:** Detection 对象数组

**示例:**
```swift
let detections = try await predictor.predict(image: inputImage)
```

##### `predict(pixelBuffer: CVPixelBuffer) async throws -> [Detection]`

对 CVPixelBuffer 进行检测（适用于实时相机流）。

**参数:**
- `pixelBuffer`: CVPixelBuffer 对象

**返回:** Detection 对象数组

### Detection 结构体

检测结果的数据结构。

```swift
public struct Detection {
    public let identifier: String      // 唯一标识符
    public let label: String          // 类别名称
    public let confidence: Float      // 置信度 (0.0-1.0)
    public let boundingBox: CGRect    // 边界框
    public let classIndex: Int        // 类别索引
}
```

### DetectionView SwiftUI 组件

用于显示检测结果的 SwiftUI 视图。

```swift
import SwiftUI

struct ContentView: View {
    @State private var detections: [Detection] = []
    @State private var inputImage: UIImage?
    
    var body: some View {
        DetectionView(
            image: inputImage,
            detections: detections,
            showLabels: true,
            showConfidence: true
        )
    }
}
```

## 📊 使用示例

### 1. 静态图片检测

```swift
import YOLOv11CoreMLSDK
import UIKit

class ImageDetectionViewController: UIViewController {
    @IBOutlet weak var imageView: UIImageView!
    
    private var predictor: YOLOv11Predictor!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        do {
            predictor = try YOLOv11Predictor()
        } catch {
            showError(error)
        }
    }
    
    @IBAction func selectImage(_ sender: UIButton) {
        let picker = UIImagePickerController()
        picker.delegate = self
        picker.sourceType = .photoLibrary
        present(picker, animated: true)
    }
    
    private func detectObjects(in image: UIImage) {
        Task {
            do {
                let detections = try await predictor.predict(image: image)
                
                await MainActor.run {
                    updateUI(with: detections, image: image)
                }
            } catch {
                await MainActor.run {
                    showError(error)
                }
            }
        }
    }
    
    private func updateUI(with detections: [Detection], image: UIImage) {
        // 在图片上绘制检测框
        let imageWithDetections = drawDetections(on: image, detections: detections)
        imageView.image = imageWithDetections
        
        // 显示检测结果
        print("检测到 \(detections.count) 个对象:")
        for detection in detections {
            print("  \(detection.label): \(String(format: "%.2f", detection.confidence))")
        }
    }
}

extension ImageDetectionViewController: UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        if let image = info[.originalImage] as? UIImage {
            imageView.image = image
            detectObjects(in: image)
        }
        picker.dismiss(animated: true)
    }
}
```

### 2. 实时相机检测

```swift
import YOLOv11CoreMLSDK
import AVFoundation
import UIKit

class CameraDetectionViewController: UIViewController {
    private var predictor: YOLOv11Predictor!
    private var captureSession: AVCaptureSession!
    private var previewLayer: AVCaptureVideoPreviewLayer!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        setupCamera()
        setupDetector()
    }
    
    private func setupDetector() {
        do {
            predictor = try YOLOv11Predictor(confidenceThreshold: 0.5)
        } catch {
            showError(error)
        }
    }
    
    private func setupCamera() {
        captureSession = AVCaptureSession()
        captureSession.sessionPreset = .medium
        
        guard let backCamera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: backCamera) else {
            return
        }
        
        captureSession.addInput(input)
        
        let videoOutput = AVCaptureVideoDataOutput()
        videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue.global(qos: .userInteractive))
        captureSession.addOutput(videoOutput)
        
        previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer.frame = view.bounds
        previewLayer.videoGravity = .resizeAspectFill
        view.layer.addSublayer(previewLayer)
        
        captureSession.startRunning()
    }
}

extension CameraDetectionViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        Task {
            do {
                let detections = try await predictor.predict(pixelBuffer: pixelBuffer)
                
                await MainActor.run {
                    updateDetectionOverlay(with: detections)
                }
            } catch {
                print("实时检测错误: \(error)")
            }
        }
    }
    
    private func updateDetectionOverlay(with detections: [Detection]) {
        // 更新检测结果覆盖层
        // 这里可以添加自定义的覆盖视图显示检测框
    }
}
```

### 3. SwiftUI 集成

```swift
import SwiftUI
import YOLOv11CoreMLSDK

struct DetectionApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

struct ContentView: View {
    @State private var selectedImage: UIImage?
    @State private var detections: [Detection] = []
    @State private var isDetecting = false
    
    private let predictor: YOLOv11Predictor
    
    init() {
        do {
            predictor = try YOLOv11Predictor()
        } catch {
            fatalError("无法初始化检测器: \(error)")
        }
    }
    
    var body: some View {
        NavigationView {
            VStack {
                if let image = selectedImage {
                    DetectionView(
                        image: image,
                        detections: detections,
                        showLabels: true,
                        showConfidence: true
                    )
                    .frame(height: 300)
                } else {
                    RoundedRectangle(cornerRadius: 12)
                        .fill(Color.gray.opacity(0.3))
                        .frame(height: 300)
                        .overlay(
                            Text("选择图片")
                                .foregroundColor(.gray)
                        )
                }
                
                Button("选择图片") {
                    // 实现图片选择逻辑
                }
                .buttonStyle(.borderedProminent)
                
                if isDetecting {
                    ProgressView("检测中...")
                        .padding()
                }
                
                List(detections, id: \.identifier) { detection in
                    HStack {
                        Text(detection.label)
                            .font(.headline)
                        Spacer()
                        Text(String(format: "%.2f", detection.confidence))
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
            }
            .navigationTitle("目标检测")
        }
    }
    
    private func detectObjects(in image: UIImage) {
        isDetecting = true
        
        Task {
            do {
                let newDetections = try await predictor.predict(image: image)
                
                await MainActor.run {
                    detections = newDetections
                    isDetecting = false
                }
            } catch {
                await MainActor.run {
                    isDetecting = false
                    print("检测失败: \(error)")
                }
            }
        }
    }
}
```

## 🔧 高级功能

### 自定义模型

```swift
// 使用自定义模型文件
let customPredictor = try YOLOv11Predictor(
    modelName: "custom_yolo11s",  // 不包含扩展名
    confidenceThreshold: 0.3,
    iouThreshold: 0.5
)
```

### 性能监控

```swift
// 启用性能监控
let predictor = try YOLOv11Predictor()

// 检测并获取性能指标
let startTime = CFAbsoluteTimeGetCurrent()
let detections = try await predictor.predict(image: image)
let inferenceTime = CFAbsoluteTimeGetCurrent() - startTime

print("推理时间: \(String(format: "%.3f", inferenceTime))s")
print("FPS: \(String(format: "%.1f", 1.0 / inferenceTime))")
```

### 批量处理

```swift
func processBatch(images: [UIImage]) async throws -> [[Detection]] {
    var results: [[Detection]] = []
    
    for image in images {
        let detections = try await predictor.predict(image: image)
        results.append(detections)
    }
    
    return results
}
```

## 🐛 常见问题

### 1. 模型加载失败

**错误:** `Failed to load CoreML model`

**解决方案:**
- 确保模型文件正确添加到 Bundle
- 检查模型文件路径和名称
- 验证 iOS/macOS 版本兼容性

### 2. 内存使用过高

**解决方案:**
```swift
// 在不需要时释放预测器
predictor = nil

// 或者使用弱引用
weak var weakPredictor = predictor
```

### 3. 实时检测性能差

**解决方案:**
- 降低相机分辨率
- 增加检测间隔
- 在后台队列执行检测

```swift
private let detectionQueue = DispatchQueue(label: "detection", qos: .userInteractive)

func optimizedDetection(pixelBuffer: CVPixelBuffer) {
    detectionQueue.async {
        // 执行检测
    }
}
```

## 📈 性能优化

### 模型预热

```swift
// 在应用启动时预热模型
class AppDelegate: UIResponder, UIApplicationDelegate {
    func application(_ application: UIApplication, didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?) -> Bool {
        
        // 预热模型
        Task {
            do {
                let predictor = try YOLOv11Predictor()
                let dummyImage = UIImage(systemName: "photo")!
                _ = try await predictor.predict(image: dummyImage)
            } catch {
                print("模型预热失败: \(error)")
            }
        }
        
        return true
    }
}
```

### 内存管理

```swift
// 使用自动释放池
func processImages(_ images: [UIImage]) async {
    for image in images {
        await withTaskGroup(of: Void.self) { group in
            group.addTask {
                autoreleasepool {
                    // 处理单张图片
                    Task {
                        let _ = try await self.predictor.predict(image: image)
                    }
                }
            }
        }
    }
}
```

## 🧪 测试

### 单元测试

```bash
# 运行所有测试
swift test

# 运行特定测试
swift test --filter YOLOv11CoreMLSDKTests
```

### 集成测试

```swift
import XCTest
@testable import YOLOv11CoreMLSDK

class IntegrationTests: XCTestCase {
    var predictor: YOLOv11Predictor!
    
    override func setUp() {
        super.setUp()
        predictor = try! YOLOv11Predictor()
    }
    
    func testImageDetection() async throws {
        let testImage = UIImage(systemName: "photo")!
        let detections = try await predictor.predict(image: testImage)
        
        XCTAssertNotNil(detections)
        // 添加更多断言
    }
}
```

## 📦 发布

### Swift Package Manager

```swift
// Package.swift
let package = Package(
    name: "YOLOv11CoreMLSDK",
    platforms: [
        .iOS(.v15),
        .macOS(.v12)
    ],
    products: [
        .library(
            name: "YOLOv11CoreMLSDK",
            targets: ["YOLOv11CoreMLSDK"]
        )
    ],
    targets: [
        .target(
            name: "YOLOv11CoreMLSDK",
            resources: [.process("Resources")]
        ),
        .testTarget(
            name: "YOLOv11CoreMLSDKTests",
            dependencies: ["YOLOv11CoreMLSDK"]
        )
    ]
)
```

## 🎉 完成

恭喜！你已经完成了完整的 YOLOv11 CoreML 项目：

1. ✅ **PyTorch 环境搭建** - 验证了原始模型功能
2. ✅ **CoreML 转换** - 成功转换并验证了模型精度
3. ✅ **Python SDK** - 创建了易用的 Python 接口
4. ✅ **Swift SDK** - 提供了原生 iOS/macOS 支持

你现在拥有了一个完整的、生产就绪的目标检测解决方案！

## 📞 支持

如遇问题，请参考：
- [项目主文档](../README.md)
- [GitHub Issues](https://github.com/your-repo/issues)
- [API 文档](https://your-docs-site.com)