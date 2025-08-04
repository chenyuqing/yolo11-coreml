import Vision
import CoreML
import UIKit

/// YOLO 错误类型
public enum YOLOError: LocalizedError {
    case modelNotFound(String)
    case modelLoadFailed(String)
    case predictionFailed(String)
    
    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let modelName):
            return "找不到模型文件: \(modelName)"
        case .modelLoadFailed(let reason):
            return "模型加载失败: \(reason)"
        case .predictionFailed(let reason):
            return "预测失败: \(reason)"
        }
    }
}

/// 检测结果结构体
/// 包含检测到的对象的所有信息
public struct Detection: Identifiable {
    /// 唯一标识符
    public let id = UUID()
    /// 对象类别标签
    public let label: String
    /// 检测置信度 (0.0 - 1.0)
    public let confidence: Float
    /// 边界框（相对坐标，范围 0.0 - 1.0）
    public let boundingBox: CGRect
    
    /// 初始化检测结果
    /// - Parameters:
    ///   - label: 对象类别标签
    ///   - confidence: 置信度
    ///   - boundingBox: 边界框
    public init(label: String, confidence: Float, boundingBox: CGRect) {
        self.label = label
        self.confidence = confidence
        self.boundingBox = boundingBox
    }
    
    /// 获取绝对坐标的边界框
    /// - Parameter imageSize: 图片尺寸
    /// - Returns: 绝对坐标的 CGRect
    public func absoluteBoundingBox(for imageSize: CGSize) -> CGRect {
        return CGRect(
            x: boundingBox.origin.x * imageSize.width,
            y: boundingBox.origin.y * imageSize.height,
            width: boundingBox.size.width * imageSize.width,
            height: boundingBox.size.height * imageSize.height
        )
    }
    
    /// 格式化的描述字符串
    public var description: String {
        return "\(label) (\(String(format: "%.2f", confidence)))"
    }
}

/// YOLOv11 预测器类
/// 处理 CoreML 模型的加载和推理
@available(iOS 15.0, macOS 12.0, *)
class YOLOv11Predictor {
    
    private let model: VNCoreMLModel
    private let modelName: String

    /// 初始化预测器
    /// - Parameter modelName: 模型名称（不包含扩展名）
    /// - Throws: 如果模型加载失败则抛出错误
    init(modelName: String = "yolo11n") throws {
        self.modelName = modelName
        
        // 尝试多种方式查找模型
        var modelURL: URL?
        
        // 1. 尝试从 Bundle 中查找 .mlmodelc 文件
        if let bundleURL = Bundle.module.url(forResource: modelName, withExtension: "mlmodelc") {
            modelURL = bundleURL
        }
        // 2. 尝试从 Bundle 中查找 .mlpackage 文件
        else if let packageURL = Bundle.module.url(forResource: modelName, withExtension: "mlpackage") {
            modelURL = packageURL
        }
        // 3. 尝试从项目资源中查找
        else if let resourceURL = Bundle.module.url(forResource: "\(modelName).mlpackage", withExtension: nil) {
            modelURL = resourceURL
        }
        
        guard let finalURL = modelURL else {
            throw YOLOError.modelNotFound(modelName)
        }
        
        // 加载 CoreML 模型
        guard let coreMLModel = try? MLModel(contentsOf: finalURL) else {
            throw YOLOError.modelLoadFailed("无法从 \(finalURL.lastPathComponent) 加载 CoreML 模型")
        }
        
        // 创建 Vision 模型
        guard let visionModel = try? VNCoreMLModel(for: coreMLModel) else {
            throw YOLOError.modelLoadFailed("无法创建 VNCoreMLModel")
        }
        
        self.model = visionModel
        
        print("✅ 成功加载模型: \(finalURL.lastPathComponent)")
    }

    /// 对给定图片执行预测
    /// - Parameter image: 输入的 CGImage
    /// - Returns: 检测结果数组
    func performPrediction(on image: CGImage) async -> [Detection] {
        let request = VNCoreMLRequest(model: model)
        request.imageCropAndScaleOption = .scaleFill

        let requestHandler = VNImageRequestHandler(cgImage: image, options: [:])

        return await withCheckedContinuation { continuation in
            request.completionHandler = { (request, error) in
                if let error = error {
                    print("⚠️ Vision 请求失败: \(error.localizedDescription)")
                    continuation.resume(returning: [])
                    return
                }

                guard let observations = request.results as? [VNRecognizedObjectObservation] else {
                    print("⚠️ 无法解析检测结果")
                    continuation.resume(returning: [])
                    return
                }

                let detections = observations.compactMap { observation -> Detection? in
                    guard let bestLabel = observation.labels.first else {
                        return nil
                    }
                    
                    return Detection(
                        label: bestLabel.identifier,
                        confidence: bestLabel.confidence,
                        boundingBox: observation.boundingBox
                    )
                }
                
                continuation.resume(returning: detections)
            }

            do {
                try requestHandler.perform([request])
            } catch {
                print("⚠️ 执行 Vision 请求失败: \(error.localizedDescription)")
                continuation.resume(returning: [])
            }
        }
    }
    
    /// 获取模型信息
    /// - Returns: 模型信息字典
    func getModelInfo() -> [String: Any] {
        return [
            "modelName": modelName,
            "modelType": "YOLOv11 CoreML",
            "visionFramework": "iOS 15.0+"
        ]
    }
}