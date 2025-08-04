
import SwiftUI
import PhotosUI

// A simple view to draw the bounding box overlays.
struct BoundingBoxOverlay: View {
    let detections: [Detection]

    var body: some View {
        GeometryReader { geometry in
            ForEach(detections, id: \.id) { detection in
                let rect = VNImageRectForNormalizedRect(detection.boundingBox, Int(geometry.size.width), Int(geometry.size.height))
                
                Rectangle()
                    .stroke(Color.red, lineWidth: 2)
                    .frame(width: rect.width, height: rect.height)
                    .position(x: rect.midX, y: rect.midY)

                Text("\(detection.label) (\(detection.confidence, specifier: "%.2f"))")
                    .font(.caption)
                    .foregroundColor(.white)
                    .padding(2)
                    .background(Color.red)
                    .position(x: rect.midX, y: rect.minY - 10)
            }
        }
    }
}

// The main view for the detection example.
public struct DetectionView: View {
    
    @StateObject private var detector = ObjectDetector()
    @State private var photosPickerItem: PhotosPickerItem?

    public init() {}

    public var body: some View {
        NavigationStack {
            VStack {
                // Image with Bounding Boxes
                ZStack {
                    if let image = detector.selectedImage {
                        Image(uiImage: image)
                            .resizable()
                            .scaledToFit()
                            .overlay(BoundingBoxOverlay(detections: detector.detections))
                    } else {
                        ContentUnavailableView("No Image Selected", systemImage: "photo.on.rectangle")
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)

                // Loading Indicator
                if detector.isLoading {
                    ProgressView("Detecting objects...")
                        .padding()
                }

                // Photos Picker
                PhotosPicker("Select Image", selection: $photosPickerItem, matching: .images)
                    .buttonStyle(.borderedProminent)
                    .padding()
            }
            .navigationTitle("YOLOv11 Detector")
            .onChange(of: photosPickerItem) { newItem in
                Task {
                    if let data = try? await newItem?.loadTransferable(type: Data.self) {
                        detector.selectedImage = UIImage(data: data)
                        detector.processImage()
                    }
                }
            }
        }
    }
}
