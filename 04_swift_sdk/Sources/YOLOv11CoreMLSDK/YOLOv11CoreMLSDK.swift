import Foundation
import Vision
import CoreML
import UIKit
import SwiftUI

/// YOLOv11 CoreML SDK 的主要接口
/// 提供简单易用的目标检测功能
@available(iOS 15.0, macOS 12.0, *)
public class YOLOv11CoreMLSDK {
    
    // MARK: - Properties
    
    private let predictor: YOLOv11Predictor
    
    /// SDK 版本
    public static let version = "1.0.0"
    
    /// 支持的最低 iOS 版本
    public static let minimumIOSVersion = "15.0"
    
    /// 支持的最低 macOS 版本
    public static let minimumMacOSVersion = "12.0"
    
    // MARK: - Initialization
    
    /// 初始化 SDK
    /// - Parameter modelName: 模型名称，默认为 "yolo11n"
    /// - Throws: 如果模型加载失败则抛出错误
    public init(modelName: String = "yolo11n") throws {
        self.predictor = try YOLOv11Predictor(modelName: modelName)
    }
    
    // MARK: - Public Methods
    
    /// 对图片进行目标检测
    /// - Parameter image: 输入的 CGImage
    /// - Returns: 检测结果数组
    public func detect(image: CGImage) async -> [Detection] {
        return await predictor.performPrediction(on: image)
    }
    
    /// 对 UIImage 进行目标检测
    /// - Parameter uiImage: 输入的 UIImage
    /// - Returns: 检测结果数组
    public func detect(uiImage: UIImage) async -> [Detection] {
        guard let cgImage = uiImage.cgImage else {
            print("⚠️ 无法从 UIImage 获取 CGImage")
            return []
        }
        return await detect(image: cgImage)
    }
    
    /// 获取 SDK 信息
    /// - Returns: SDK 信息字典
    public func getSDKInfo() -> [String: Any] {
        return [
            "version": Self.version,
            "minimumIOSVersion": Self.minimumIOSVersion,
            "minimumMacOSVersion": Self.minimumMacOSVersion,
            "modelSupported": true,
            "coreMLVersion": "iOS 15.0+"
        ]
    }
    
    /// 性能基准测试
    /// - Parameters:
    ///   - image: 测试图片
    ///   - iterations: 测试次数
    /// - Returns: 性能统计信息
    public func benchmark(image: CGImage, iterations: Int = 10) async -> BenchmarkResult {
        var times: [TimeInterval] = []
        
        // 预热
        _ = await detect(image: image)
        _ = await detect(image: image)
        
        // 基准测试
        for _ in 0..<iterations {
            let startTime = CFAbsoluteTimeGetCurrent()
            _ = await detect(image: image)
            let endTime = CFAbsoluteTimeGetCurrent()
            times.append(endTime - startTime)
        }
        
        let avgTime = times.reduce(0, +) / Double(times.count)
        let minTime = times.min() ?? 0
        let maxTime = times.max() ?? 0
        let fps = 1.0 / avgTime
        
        return BenchmarkResult(
            iterations: iterations,
            averageTime: avgTime,
            minTime: minTime,
            maxTime: maxTime,
            averageFPS: fps
        )
    }
}

/// 性能基准测试结果
public struct BenchmarkResult {
    /// 测试次数
    public let iterations: Int
    /// 平均时间（秒）
    public let averageTime: TimeInterval
    /// 最短时间（秒）
    public let minTime: TimeInterval
    /// 最长时间（秒）
    public let maxTime: TimeInterval
    /// 平均 FPS
    public let averageFPS: Double
    
    /// 格式化的描述字符串
    public var description: String {
        return """
        基准测试结果 (共 \(iterations) 次):
        - 平均时间: \(String(format: "%.3f", averageTime))s
        - 最短时间: \(String(format: "%.3f", minTime))s
        - 最长时间: \(String(format: "%.3f", maxTime))s
        - 平均 FPS: \(String(format: "%.1f", averageFPS))
        """
    }
}