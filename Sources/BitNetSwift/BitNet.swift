import BitNet
import Foundation

public final class BitNet {
    public init() {
        ggml_bitnet_init()
    }

    deinit {
        ggml_bitnet_free()
    }

    /// Returns true if matrix multiplication is supported for the given tensors.
    public func canMultiply(
        _ src0: GGMLTensor,
        _ src1: GGMLTensor,
        _ dst: GGMLTensor
    ) -> Bool {
        ggml_bitnet_can_mul_mat(src0.pointer, src1.pointer, dst.pointer)
    }

    /// Returns the workspace size required for matrix multiplication.
    public func workspaceSizeForMultiplication(
        src0: GGMLTensor,
        src1: GGMLTensor,
        dst: GGMLTensor
    ) -> Int {
        Int(ggml_bitnet_mul_mat_get_wsize(src0.pointer, src1.pointer, dst.pointer))
    }

    /// Initializes a matrix multiplication task.
    public func initializeMultiplicationTask(
        src1: [UInt8],
        qlut: [UInt8],
        lutScales: [Float],
        lutBiases: [Float],
        n: Int, k: Int, m: Int, bits: Int
    ) {
        src1.withUnsafeBufferPointer { src1Ptr in
            qlut.withUnsafeBufferPointer { qlutPtr in
                lutScales.withUnsafeBufferPointer { lutScalesPtr in
                    lutBiases.withUnsafeBufferPointer { lutBiasesPtr in
                        ggml_bitnet_mul_mat_task_init(
                            UnsafeMutableRawPointer(mutating: src1Ptr.baseAddress),
                            UnsafeMutableRawPointer(mutating: qlutPtr.baseAddress),
                            UnsafeMutableRawPointer(mutating: lutScalesPtr.baseAddress),
                            UnsafeMutableRawPointer(mutating: lutBiasesPtr.baseAddress),
                            Int32(n), Int32(k), Int32(m), Int32(bits)
                        )
                    }
                }
            }
        }
    }

    /// Computes a matrix multiplication task.
    public func computeMultiplicationTask(
        src0: [UInt8],
        scales: [Float],
        qlut: [UInt8],
        lutScales: [Float],
        lutBiases: [Float],
        dst: inout [Float],
        n: Int, k: Int, m: Int, bits: Int
    ) {
        src0.withUnsafeBufferPointer { src0Ptr in
            scales.withUnsafeBufferPointer { scalesPtr in
                qlut.withUnsafeBufferPointer { qlutPtr in
                    lutScales.withUnsafeBufferPointer { lutScalesPtr in
                        lutBiases.withUnsafeBufferPointer { lutBiasesPtr in
                            dst.withUnsafeMutableBufferPointer { dstPtr in
                                ggml_bitnet_mul_mat_task_compute(
                                    UnsafeMutableRawPointer(mutating: src0Ptr.baseAddress),
                                    UnsafeMutableRawPointer(mutating: scalesPtr.baseAddress),
                                    UnsafeMutableRawPointer(mutating: qlutPtr.baseAddress),
                                    UnsafeMutableRawPointer(mutating: lutScalesPtr.baseAddress),
                                    UnsafeMutableRawPointer(mutating: lutBiasesPtr.baseAddress),
                                    dstPtr.baseAddress,
                                    Int32(n), Int32(k), Int32(m), Int32(bits)
                                )
                            }
                        }
                    }
                }
            }
        }
    }

    /// Transforms a tensor for BitNet.
    public func transform(_ tensor: GGMLTensor) {
        ggml_bitnet_transform_tensor(tensor.mutablePointer)
    }

    /// Returns the number of bits for a given tensor type.
    public func bits(for type: ggml_type) -> Int {
        ggml_bitnet_get_type_bits(type)
    }

    /// Sets the number of threads for BitNet operations.
    public func setThreadCount(_ count: Int) {
        ggml_bitnet_set_n_threads(Int32(count))
    }

    #if arch(arm64) || arch(arm)
    /// ARM-specific: Quantized GEMM with lookup table.
    public func quantizedGemmLUT(
        m: Int, k: Int,
        A: [Float],
        lut: [UInt8],
        scales: [Float],
        lutScales: [Float],
        c: inout [Float]
    ) {
        A.withUnsafeBufferPointer { aPtr in
            lut.withUnsafeBufferPointer { lutPtr in
                scales.withUnsafeBufferPointer { scalesPtr in
                    lutScales.withUnsafeBufferPointer { lutScalesPtr in
                        c.withUnsafeMutableBufferPointer { cPtr in
                            ggml_qgemm_lut(
                                Int32(m), Int32(k),
                                UnsafeMutableRawPointer(mutating: aPtr.baseAddress),
                                UnsafeMutableRawPointer(mutating: lutPtr.baseAddress),
                                UnsafeMutableRawPointer(mutating: scalesPtr.baseAddress),
                                UnsafeMutableRawPointer(mutating: lutScalesPtr.baseAddress),
                                cPtr.baseAddress
                            )
                        }
                    }
                }
            }
        }
    }

    /// ARM-specific: Preprocess for quantized GEMM.
    public func preprocessQuantizedGemm(
        m: Int, k: Int,
        b: [Float],
        lutScales: [Float],
        qlut: inout [UInt8]
    ) {
        b.withUnsafeBufferPointer { bPtr in
            lutScales.withUnsafeBufferPointer { lutScalesPtr in
                qlut.withUnsafeMutableBufferPointer { qlutPtr in
                    ggml_preprocessor(
                        Int32(m), Int32(k),
                        UnsafeMutableRawPointer(mutating: bPtr.baseAddress),
                        UnsafeMutableRawPointer(mutating: lutScalesPtr.baseAddress),
                        qlutPtr.baseAddress
                    )
                }
            }
        }
    }
    #endif

    #if arch(x86_64) || arch(i386)
    /// x86-specific: Quantized GEMM with lookup table.
    public func quantizedGemmLUT(
        batchSize: Int, m: Int, k: Int, blockK: Int,
        a: [Float],
        sign: [UInt8],
        lut: [UInt8],
        scales: [Float],
        lutScales: [Float],
        c: inout [Float]
    ) {
        a.withUnsafeBufferPointer { aPtr in
            sign.withUnsafeBufferPointer { signPtr in
                lut.withUnsafeBufferPointer { lutPtr in
                    scales.withUnsafeBufferPointer { scalesPtr in
                        lutScales.withUnsafeBufferPointer { lutScalesPtr in
                            c.withUnsafeMutableBufferPointer { cPtr in
                                ggml_qgemm_lut(
                                    Int32(batchSize), Int32(m), Int32(k), Int32(blockK),
                                    UnsafeMutableRawPointer(mutating: aPtr.baseAddress),
                                    UnsafeMutableRawPointer(mutating: signPtr.baseAddress),
                                    UnsafeMutableRawPointer(mutating: lutPtr.baseAddress),
                                    UnsafeMutableRawPointer(mutating: scalesPtr.baseAddress),
                                    UnsafeMutableRawPointer(mutating: lutScalesPtr.baseAddress),
                                    cPtr.baseAddress
                                )
                            }
                        }
                    }
                }
            }
        }
    }

    /// x86-specific: Preprocess for quantized GEMM.
    public func preprocessQuantizedGemm(
        batchSize: Int, m: Int, threeK: Int, twoK: Int,
        b: [Float],
        lutScales: [Float],
        threeQLUT: inout [UInt8],
        twoQLUT: inout [UInt8]
    ) {
        b.withUnsafeBufferPointer { bPtr in
            lutScales.withUnsafeBufferPointer { lutScalesPtr in
                threeQLUT.withUnsafeMutableBufferPointer { threeQLUTPtr in
                    twoQLUT.withUnsafeMutableBufferPointer { twoQLUTPtr in
                        ggml_preprocessor(
                            Int32(batchSize), Int32(m), Int32(threeK), Int32(twoK),
                            UnsafeMutableRawPointer(mutating: bPtr.baseAddress),
                            UnsafeMutableRawPointer(mutating: lutScalesPtr.baseAddress),
                            threeQLUTPtr.baseAddress,
                            twoQLUTPtr.baseAddress
                        )
                    }
                }
            }
        }
    }
    #endif
}
