import Metal
import Foundation
import simd

typealias Float4 = SIMD4<Float>

struct SimpleFileWriter {
  var fileHandle: FileHandle
  
  init(filePath: String, append: Bool = false) {
    if !FileManager.default.fileExists(atPath: filePath) {
      FileManager.default.createFile(atPath: filePath, contents: nil, attributes: nil)
    }
    
    fileHandle = FileHandle(forWritingAtPath: filePath)!
    if !append {
      fileHandle.truncateFile(atOffset: 0)
    }
  }
  
  func write(content: String) {
    fileHandle.seekToEndOfFile()
    guard let data = content.data(using: String.Encoding.ascii) else {
      fatalError("Could not write \(content) to file!")
    }
    fileHandle.write(data)
  }
}

var imageWidth = 480
var imageHeight = 270

let device = MTLCreateSystemDefaultDevice()!
let library = try! device.makeDefaultLibrary(bundle: Bundle.module)
let primaryRayFunc = library.makeFunction(name: "primary_ray")!
let pipeline = try! device.makeComputePipelineState(function: primaryRayFunc)

var pixelData: [Float4] = (0..<(imageWidth * imageHeight)).map{ _ in Float4(0, 0, 0, 0)}
var pixelCount = UInt(pixelData.count)

let pixelDataBuffer = device.makeBuffer(bytes: &pixelData, length: Int(pixelCount) * MemoryLayout<Float4>.stride, options: [])!
let pixelDataMirrorPointer = pixelDataBuffer.contents().bindMemory(to: Float4.self, capacity: Int(pixelCount))
let pixelDataMirrorBuffer = UnsafeBufferPointer(start: pixelDataMirrorPointer, count: Int(pixelCount))

let commandQueue = device.makeCommandQueue()!
let commandBuffer = commandQueue.makeCommandBuffer()!
let commandEncoder = commandBuffer.makeComputeCommandEncoder()!

commandEncoder.setComputePipelineState(pipeline)
commandEncoder.setBuffer(pixelDataBuffer, offset: 0, index: 0)
commandEncoder.setBytes(&pixelCount, length: MemoryLayout<Int>.stride, index: 1)
commandEncoder.setBytes(&imageWidth, length: MemoryLayout<Int>.stride, index: 2)
commandEncoder.setBytes(&imageHeight, length: MemoryLayout<Int>.stride, index: 3)

// We have to calculate the sum `pixelCount` times
// => amount of threadgroups is `resultsCount` / `threadExecutionWidth` (rounded up)
// because each threadgroup will process `threadExecutionWidth` threads
let threadgroupsPerGrid = MTLSize(width: (Int(pixelCount) + pipeline.threadExecutionWidth - 1) / pipeline.threadExecutionWidth, height: 1, depth: 1)
// Here we set that each threadgroup should process `threadExecutionWidth` threads
// the only important thing for performance is that this number is a multiple of
// `threadExecutionWidth` (here 1 times)
let threadsPerThreadgroup = MTLSize(width: pipeline.threadExecutionWidth, height: 1, depth: 1)
commandEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
commandEncoder.endEncoding()

commandBuffer.commit()
commandBuffer.waitUntilCompleted()

let sfw = SimpleFileWriter(filePath: "/Users/pprovins/Desktop/here.ppm")
sfw.write(content: "P3\n")
sfw.write(content: "\(imageWidth) \(imageHeight)\n")
sfw.write(content: "255\n")

for pixel in pixelDataMirrorBuffer {
  sfw.write(content: "\(UInt8(pixel.x * 255)) \(UInt8(pixel.y * 255)) \(UInt8(pixel.z * 255)) ")
}

sfw.write(content: "\n")
