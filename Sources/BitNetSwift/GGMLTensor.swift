import BitNet

public struct GGMLTensor {
    public let pointer: UnsafePointer<ggml_tensor>
    public var mutablePointer: UnsafeMutablePointer<ggml_tensor> {
        UnsafeMutablePointer(mutating: pointer)
    }

    public init(pointer: UnsafePointer<ggml_tensor>) {
        self.pointer = pointer
    }

    public init(mutablePointer: UnsafeMutablePointer<ggml_tensor>) {
        self.pointer = UnsafePointer(mutablePointer)
    }

    // Swift-friendly initializer for Float arrays
    public init?(from array: [Float], shape: [Int], context: GGMLContext) {
        guard let tensor = ggml_new_tensor_1d(context.pointer, GGML_TYPE_F32, Int32(array.count)) else {
            return nil
        }
        let tensorData = ggml_get_data(tensor)
        array.withUnsafeBytes { srcPtr in
            memcpy(tensorData, srcPtr.baseAddress, array.count * MemoryLayout<Float>.stride)
        }
        self.pointer = UnsafePointer(tensor)
    }

    // Swift-friendly initializer for UInt8 arrays
    public init?(from array: [UInt8], shape: [Int], context: GGMLContext) {
        guard let tensor = ggml_new_tensor_1d(context.pointer, GGML_TYPE_I8, Int32(array.count)) else {
            return nil
        }
        let tensorData = ggml_get_data(tensor)
        array.withUnsafeBytes { srcPtr in
            memcpy(tensorData, srcPtr.baseAddress, array.count * MemoryLayout<UInt8>.stride)
        }
        self.pointer = UnsafePointer(tensor)
    }
}
