import Metal
import Foundation
import simd

typealias Float4 = SIMD4<Float>
typealias Float3 = SIMD3<Float>

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
var pixelCount = UInt(imageWidth * imageHeight)
var sampleCount = 150
var bounceCount = 150
var cameraOrigin = Float3(repeating: 0.0)

// MARK: Setup Device and Library
let device = MTLCreateSystemDefaultDevice()!
let library = try! device.makeDefaultLibrary(bundle: Bundle.module)
let spawnRayFunc = library.makeFunction(name: "spawn_rays")!
let traceRayFunc = library.makeFunction(name: "ray_trace")!
let combineFunc = library.makeFunction(name: "combine_results")!

let spawnPipeline = try! device.makeComputePipelineState(function: spawnRayFunc)
let tracePipeline = try! device.makeComputePipelineState(function: traceRayFunc)
let combinePipeline = try! device.makeComputePipelineState(function: combineFunc)

// MARK: Setup Command Enqueuing
let commandQueue = device.makeCommandQueue()!
let commandBufferDescriptor = MTLCommandBufferDescriptor()
commandBufferDescriptor.errorOptions = MTLCommandBufferErrorOption.encoderExecutionStatus
let commandBuffer = commandQueue.makeCommandBuffer(descriptor: commandBufferDescriptor)!
let commandEncoder = commandBuffer.makeComputeCommandEncoder()!

// MARK: Ray Direction Creation

/*
 Setup the ray directions. For pixel, there are sampleCount samples.
 Each sample will have a slightly different direction.
 */
var directionData: [Float3] = (0..<(Int(pixelCount) * sampleCount)).map{ _ in Float3(0, 0, 0)}
var directionCount = UInt(directionData.count)

let directionDataBuffer = device.makeBuffer(bytes: &directionData, length: Int(directionCount) * MemoryLayout<Float3>.stride, options: [])!
let directionDataMirrorPointer = directionDataBuffer.contents().bindMemory(to: Float3.self, capacity: Int(directionCount))
let directionDataMirrorBuffer = UnsafeBufferPointer(start: directionDataMirrorPointer, count: Int(directionCount))

commandEncoder.setComputePipelineState(spawnPipeline)
commandEncoder.setBytes(&cameraOrigin, length: MemoryLayout<Float3>.stride, index: 0)
commandEncoder.setBuffer(directionDataBuffer, offset: 0, index: 1)
commandEncoder.setBytes(&imageWidth, length: MemoryLayout<Int>.stride, index: 2)
commandEncoder.setBytes(&imageHeight, length: MemoryLayout<Int>.stride, index: 3)
commandEncoder.setBytes(&sampleCount, length: MemoryLayout<Int>.stride, index: 4)

// We have to calculate the sum `pixelCount` times
// => amount of threadgroups is `resultsCount` / `threadExecutionWidth` (rounded up)
// because each threadgroup will process `threadExecutionWidth` threads
var threadExecutionWidth = spawnPipeline.maxTotalThreadsPerThreadgroup
var threadgroupsPerGrid = MTLSize(width: (Int(pixelCount) + threadExecutionWidth - 1) / threadExecutionWidth, height: 1, depth: 1)
// Here we set that each threadgroup should process `threadExecutionWidth` threads
// the only important thing for performance is that this number is a multiple of
// `threadExecutionWidth` (here 1 times)
var threadsPerThreadgroup = MTLSize(width: threadExecutionWidth, height: 1, depth: 1)
commandEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)

commandEncoder.setComputePipelineState(tracePipeline)
commandEncoder.setBuffer(directionDataBuffer, offset: 0, index: 0)
commandEncoder.setBytes(&cameraOrigin, length: MemoryLayout<Float3>.stride, index: 1)
commandEncoder.setBytes(&imageWidth, length: MemoryLayout<Int>.stride, index: 2)
commandEncoder.setBytes(&imageHeight, length: MemoryLayout<Int>.stride, index: 3)
commandEncoder.setBytes(&sampleCount, length: MemoryLayout<Int>.stride, index: 4)
commandEncoder.setBytes(&bounceCount, length: MemoryLayout<Int>.stride, index: 5)


// We have to calculate the sum `pixelCount` times
// => amount of threadgroups is `resultsCount` / `threadExecutionWidth` (rounded up)
// because each threadgroup will process `threadExecutionWidth` threads
threadExecutionWidth = tracePipeline.maxTotalThreadsPerThreadgroup
threadgroupsPerGrid = MTLSize(width: (Int(directionCount) + threadExecutionWidth - 1) / threadExecutionWidth, height: 1, depth: 1)
// Here we set that each threadgroup should process `threadExecutionWidth` threads
// the only important thing for performance is that this number is a multiple of
// `threadExecutionWidth` (here 1 times)
threadsPerThreadgroup = MTLSize(width: threadExecutionWidth, height: 1, depth: 1)
commandEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)

// MARK: Pixel Color Buffer Setup Begin
var pixelData: [Float4] = (0..<pixelCount).map{ _ in Float4(0, 0, 0, 0)}
let pixelDataBuffer = device.makeBuffer(bytes: &pixelData, length: Int(pixelCount) * MemoryLayout<Float4>.stride, options: [])!
let pixelDataMirrorPointer = pixelDataBuffer.contents().bindMemory(to: Float4.self, capacity: Int(pixelCount))
let pixelDataMirrorBuffer = UnsafeBufferPointer(start: pixelDataMirrorPointer, count: Int(pixelCount))
/*
 End Pixel Color Buffer Setup
 */

commandEncoder.setComputePipelineState(combinePipeline)
commandEncoder.setBuffer(directionDataBuffer, offset: 0, index: 0)
commandEncoder.setBuffer(pixelDataBuffer, offset: 0, index: 1)
commandEncoder.setBytes(&imageWidth, length: MemoryLayout<Int>.stride, index: 2)
commandEncoder.setBytes(&imageHeight, length: MemoryLayout<Int>.stride, index: 3)
commandEncoder.setBytes(&sampleCount, length: MemoryLayout<Int>.stride, index: 4)

// We have to calculate the sum `pixelCount` times
// => amount of threadgroups is `resultsCount` / `threadExecutionWidth` (rounded up)
// because each threadgroup will process `threadExecutionWidth` threads
threadExecutionWidth = combinePipeline.maxTotalThreadsPerThreadgroup
threadgroupsPerGrid = MTLSize(width: (Int(pixelCount) + threadExecutionWidth - 1) / threadExecutionWidth, height: 1, depth: 1)
// Here we set that each threadgroup should process `threadExecutionWidth` threads
// the only important thing for performance is that this number is a multiple of
// `threadExecutionWidth` (here 1 times)
threadsPerThreadgroup = MTLSize(width: threadExecutionWidth, height: 1, depth: 1)
commandEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)

commandEncoder.endEncoding()
commandBuffer.commit()
commandBuffer.waitUntilCompleted()

if let error = commandBuffer.error as NSError? {
  if let encoderInfo = error.userInfo[MTLCommandBufferEncoderInfoErrorKey] as? [MTLCommandBufferEncoderInfo] {
    for info in encoderInfo {
      print(info.label + info.debugSignposts.joined())
    }
  }
}

let sfw = SimpleFileWriter(filePath: "/Users/pprovins/Desktop/render.ppm")
sfw.write(content: "P3\n")
sfw.write(content: "\(imageWidth) \(imageHeight)\n")
sfw.write(content: "255\n")

for pixel in pixelDataMirrorBuffer {
  sfw.write(content: "\(UInt8(pixel.x * 255)) \(UInt8(pixel.y * 255)) \(UInt8(pixel.z * 255)) ")
}

sfw.write(content: "\n")
