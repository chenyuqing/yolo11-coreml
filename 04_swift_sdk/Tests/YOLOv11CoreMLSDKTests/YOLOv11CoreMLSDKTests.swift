import XCTest
import CoreGraphics
import UIKit
@testable import YOLOv11CoreMLSDK

/// YOLOv11 CoreML SDK 单元测试
@available(iOS 15.0, macOS 12.0, *)
final class YOLOv11CoreMLSDKTests: XCTestCase {
    
    var sdk: YOLOv11CoreMLSDK?
    
    override func setUpWithError() throws {
        try super.setUpWithError()
        
        // 尝试初始化 SDK
        do {
            sdk = try YOLOv11CoreMLSDK()
        } catch {
            // 如果模型不存在，跳过测试
            throw XCTSkip("模型文件不存在，跳过测试: \(error.localizedDescription)")
        }
    }
    
    override func tearDownWithError() throws {
        sdk = nil
        try super.tearDownWithError()
    }
    
    /// 测试 SDK 初始化
    func testSDKInitialization() throws {
        XCTAssertNotNil(sdk, "SDK 应该成功初始化")
    }
    
    /// 测试 SDK 版本信息
    func testSDKInfo() throws {
        guard let sdk = sdk else {
            XCTFail("SDK 未初始化")
            return
        }
        
        let info = sdk.getSDKInfo()
        
        XCTAssertNotNil(info["version"], "应该包含版本信息")
        XCTAssertNotNil(info["minimumIOSVersion"], "应该包含最低 iOS 版本")
        XCTAssertNotNil(info["minimumMacOSVersion"], "应该包含最低 macOS 版本")
        
        XCTAssertEqual(info["version"] as? String, YOLOv11CoreMLSDK.version)
    }
    
    /// 测试创建测试图片
    func testCreateTestImage() throws {
        let testImage = createTestImage()
        XCTAssertNotNil(testImage, "应该能创建测试图片")
        
        let size = testImage.size
        XCTAssertGreaterThan(size.width, 0, "图片宽度应该大于 0")
        XCTAssertGreaterThan(size.height, 0, "图片高度应该大于 0")
    }
    
    /// 测试图片检测功能
    func testImageDetection() async throws {
        guard let sdk = sdk else {
            XCTFail("SDK 未初始化")
            return
        }
        
        let testImage = createTestImage()
        guard let cgImage = testImage.cgImage else {
            XCTFail("无法获取 CGImage")
            return
        }
        
        let detections = await sdk.detect(image: cgImage)
        
        // 检测结果可能为空（测试图片可能没有可检测的对象）
        // 这里主要测试函数是否正常执行
        XCTAssertNotNil(detections, "检测结果不应该为 nil")
        
        print("检测到 \(detections.count) 个对象")
        
        for detection in detections {
            XCTAssertFalse(detection.label.isEmpty, "标签不应该为空")
            XCTAssertGreaterThanOrEqual(detection.confidence, 0.0, "置信度应该大于等于 0")
            XCTAssertLessThanOrEqual(detection.confidence, 1.0, "置信度应该小于等于 1")
            
            let bbox = detection.boundingBox
            XCTAssertGreaterThanOrEqual(bbox.origin.x, 0.0, "边界框 x 坐标应该大于等于 0")
            XCTAssertGreaterThanOrEqual(bbox.origin.y, 0.0, "边界框 y 坐标应该大于等于 0")
            XCTAssertLessThanOrEqual(bbox.origin.x + bbox.size.width, 1.0, "边界框右边界应该小于等于 1")
            XCTAssertLessThanOrEqual(bbox.origin.y + bbox.size.height, 1.0, "边界框下边界应该小于等于 1")
            
            print("检测结果: \(detection.description)")
        }
    }
    
    /// 测试 UIImage 检测功能
    func testUIImageDetection() async throws {
        guard let sdk = sdk else {
            XCTFail("SDK 未初始化")
            return
        }
        
        let testImage = createTestImage()
        let detections = await sdk.detect(uiImage: testImage)
        
        XCTAssertNotNil(detections, "检测结果不应该为 nil")
        print("UIImage 检测到 \(detections.count) 个对象")
    }
    
    /// 测试绝对坐标转换
    func testAbsoluteBoundingBox() throws {
        let detection = Detection(
            label: "test",
            confidence: 0.5,
            boundingBox: CGRect(x: 0.1, y: 0.2, width: 0.3, height: 0.4)
        )
        
        let imageSize = CGSize(width: 100, height: 200)
        let absoluteBBox = detection.absoluteBoundingBox(for: imageSize)
        
        XCTAssertEqual(absoluteBBox.origin.x, 10.0, accuracy: 0.01)
        XCTAssertEqual(absoluteBBox.origin.y, 40.0, accuracy: 0.01)
        XCTAssertEqual(absoluteBBox.size.width, 30.0, accuracy: 0.01)
        XCTAssertEqual(absoluteBBox.size.height, 80.0, accuracy: 0.01)
    }
    
    /// 测试性能基准
    func testBenchmark() async throws {
        guard let sdk = sdk else {
            XCTFail("SDK 未初始化")
            return
        }
        
        let testImage = createTestImage()
        guard let cgImage = testImage.cgImage else {
            XCTFail("无法获取 CGImage")
            return
        }
        
        let benchmarkResult = await sdk.benchmark(image: cgImage, iterations: 3)
        
        XCTAssertEqual(benchmarkResult.iterations, 3)
        XCTAssertGreaterThan(benchmarkResult.averageTime, 0)
        XCTAssertGreaterThan(benchmarkResult.averageFPS, 0)
        XCTAssertLessThanOrEqual(benchmarkResult.minTime, benchmarkResult.averageTime)
        XCTAssertGreaterThanOrEqual(benchmarkResult.maxTime, benchmarkResult.averageTime)
        
        print(benchmarkResult.description)
    }
    
    /// 测试错误处理
    func testErrorHandling() throws {
        // 测试不存在的模型名称
        XCTAssertThrowsError(try YOLOv11CoreMLSDK(modelName: "nonexistent_model")) { error in
            XCTAssertTrue(error is YOLOError, "应该抛出 YOLOError")
            
            if case YOLOError.modelNotFound(let modelName) = error {
                XCTAssertEqual(modelName, "nonexistent_model")
            } else {
                XCTFail("应该是 modelNotFound 错误")
            }
        }
    }
    
    // MARK: - Helper Methods
    
    /// 创建测试用的图片
    private func createTestImage() -> UIImage {
        let size = CGSize(width: 640, height: 640)
        
        UIGraphicsBeginImageContextWithOptions(size, false, 1.0)
        defer { UIGraphicsEndImageContext() }
        
        guard let context = UIGraphicsGetCurrentContext() else {
            return UIImage()
        }
        
        // 创建一个简单的测试图片
        // 绘制背景
        context.setFillColor(UIColor.white.cgColor)
        context.fill(CGRect(origin: .zero, size: size))
        
        // 绘制一些形状作为测试对象
        context.setFillColor(UIColor.red.cgColor)
        context.fillEllipse(in: CGRect(x: 100, y: 100, width: 100, height: 100))
        
        context.setFillColor(UIColor.blue.cgColor)
        context.fill(CGRect(x: 300, y: 200, width: 150, height: 100))
        
        context.setFillColor(UIColor.green.cgColor)
        context.fillEllipse(in: CGRect(x: 200, y: 400, width: 200, height: 150))
        
        return UIGraphicsGetImageFromCurrentImageContext() ?? UIImage()
    }
}