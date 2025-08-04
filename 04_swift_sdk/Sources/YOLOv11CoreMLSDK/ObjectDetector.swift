
import SwiftUI
import PhotosUI

@MainActor
public class ObjectDetector: ObservableObject {
    
    @Published public var selectedImage: UIImage? = nil
    @Published public var detections: [Detection] = []
    @Published public var isLoading: Bool = false

    private let predictor = YOLOv11Predictor()

    public init() {}

    public func processImage() {
        guard let image = selectedImage else { return }
        guard let cgImage = image.cgImage else {
            print("Failed to get CGImage from selected image.")
            return
        }

        isLoading = true
        detections = []

        Task {
            let newDetections = await predictor.performPrediction(on: cgImage)
            
            // Update properties on the main thread
            DispatchQueue.main.async {
                self.detections = newDetections
                self.isLoading = false
            }
        }
    }
}
