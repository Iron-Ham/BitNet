import BitNet

public final class GGMLContext {
    private let rawPointer: UnsafeMutablePointer<ggml_context>
    public var pointer: UnsafeMutablePointer<ggml_context> { rawPointer }

    /// Default memory size: 128MB
    public static let defaultMemorySize: Int = 128 * 1024 * 1024

    public init?(memorySize: Int = GGMLContext.defaultMemorySize) {
        var params = ggml_init_params(mem_size: memorySize, mem_buffer: nil, no_alloc: false)
        guard let ctx = ggml_init(params) else { return nil }
        self.rawPointer = ctx
    }

    deinit {
        ggml_free(rawPointer)
    }
}
