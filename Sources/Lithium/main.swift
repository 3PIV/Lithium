import Metal
import Foundation
import simd

typealias Float4 = SIMD4<Float>
typealias Float3 = SIMD3<Float>

func makeImage(for texture: MTLTexture) -> CGImage? {
  assert(texture.pixelFormat == .rgba8Unorm)
  
  let width = texture.width
  let height = texture.height
  let pixelByteCount = 4 * MemoryLayout<UInt8>.size
  let imageBytesPerRow = width * pixelByteCount
  let imageByteCount = imageBytesPerRow * height
  let imageBytes = UnsafeMutableRawPointer.allocate(byteCount: imageByteCount, alignment: pixelByteCount)
  defer {
    imageBytes.deallocate()
  }
  
  texture.getBytes(imageBytes,
                   bytesPerRow: imageBytesPerRow,
                   from: MTLRegionMake2D(0, 0, width, height),
                   mipmapLevel: 0)
    
  guard let colorSpace = CGColorSpace(name: CGColorSpace.linearSRGB) else { return nil }
  let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue
  guard let bitmapContext = CGContext(data: nil,
                                      width: width,
                                      height: height,
                                      bitsPerComponent: 8,
                                      bytesPerRow: imageBytesPerRow,
                                      space: colorSpace,
                                      bitmapInfo: bitmapInfo) else { return nil }
  bitmapContext.data?.copyMemory(from: imageBytes, byteCount: imageByteCount)
  let image = bitmapContext.makeImage()
  return image
}

func saveImage(for texture: MTLTexture, where location: String) {
  guard let accumulatedImage = makeImage(for: texture) else {
    fatalError("Could not create an image from the specified texture.")
  }
  
  guard let imageDestination = CGImageDestinationCreateWithURL(NSURL.fileURL(withPath: location) as CFURL, kUTTypePNG, 1, nil) else {
    fatalError("Could not save the image to the provided location: \(location).")
  }
  
  CGImageDestinationAddImage(imageDestination, accumulatedImage, nil)
  CGImageDestinationFinalize(imageDestination)
}

var imageWidth = 960
var imageHeight = 540
var pixelCount = UInt(imageWidth * imageHeight)
var sampleCount = 75
var bounceCount = 30
var cameraOrigin = Float3(-1.0, -0.25, 2.0)
var cameraTarget = Float3(-0.1, 0, -1.0)
var fieldOfViewDegrees: Float = 40.0
var cameraFocusDistance: Float = length(cameraOrigin - cameraTarget)
var cameraApertureRadius: Float = 0.25

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
let samplesByPixelsCount = Int(pixelCount) * sampleCount

let originDataBuffer = device.makeBuffer(length: samplesByPixelsCount * MemoryLayout<Float3>.stride, options: [.storageModePrivate])!
let directionDataBuffer = device.makeBuffer(length: samplesByPixelsCount * MemoryLayout<Float3>.stride, options: [.storageModePrivate])!

commandEncoder.setComputePipelineState(spawnPipeline)
commandEncoder.setBytes(&cameraOrigin, length: MemoryLayout<Float3>.stride, index: 0)
commandEncoder.setBytes(&cameraTarget, length: MemoryLayout<Float3>.stride, index: 1)
commandEncoder.setBytes(&fieldOfViewDegrees, length: MemoryLayout<Float>.stride, index: 2)
commandEncoder.setBytes(&cameraFocusDistance, length: MemoryLayout<Float>.stride, index: 3)
commandEncoder.setBytes(&cameraApertureRadius, length: MemoryLayout<Float>.stride, index: 4)
commandEncoder.setBuffer(originDataBuffer, offset: 0, index: 5)
commandEncoder.setBuffer(directionDataBuffer, offset: 0, index: 6)
commandEncoder.setBytes(&imageWidth, length: MemoryLayout<Int>.stride, index: 7)
commandEncoder.setBytes(&imageHeight, length: MemoryLayout<Int>.stride, index: 8)
commandEncoder.setBytes(&sampleCount, length: MemoryLayout<Int>.stride, index: 9)

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
commandEncoder.setBuffer(originDataBuffer, offset: 0, index: 0)
commandEncoder.setBuffer(directionDataBuffer, offset: 0, index: 1)
commandEncoder.setBytes(&imageWidth, length: MemoryLayout<Int>.stride, index: 2)
commandEncoder.setBytes(&imageHeight, length: MemoryLayout<Int>.stride, index: 3)
commandEncoder.setBytes(&sampleCount, length: MemoryLayout<Int>.stride, index: 4)
commandEncoder.setBytes(&bounceCount, length: MemoryLayout<Int>.stride, index: 5)


// We have to calculate the sum `pixelCount` times
// => amount of threadgroups is `resultsCount` / `threadExecutionWidth` (rounded up)
// because each threadgroup will process `threadExecutionWidth` threads
threadExecutionWidth = tracePipeline.maxTotalThreadsPerThreadgroup
threadgroupsPerGrid = MTLSize(width: (Int(samplesByPixelsCount) + threadExecutionWidth - 1) / threadExecutionWidth, height: 1, depth: 1)
// Here we set that each threadgroup should process `threadExecutionWidth` threads
// the only important thing for performance is that this number is a multiple of
// `threadExecutionWidth` (here 1 times)
threadsPerThreadgroup = MTLSize(width: threadExecutionWidth, height: 1, depth: 1)
commandEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)

// MARK: Pixel Color Buffer Setup Begin
let accumulantTextureDescriptor = MTLTextureDescriptor()
accumulantTextureDescriptor.width = imageWidth
accumulantTextureDescriptor.height = imageHeight
accumulantTextureDescriptor.depth = 1
accumulantTextureDescriptor.usage = .shaderWrite
accumulantTextureDescriptor.mipmapLevelCount = 1
accumulantTextureDescriptor.allowGPUOptimizedContents = false
accumulantTextureDescriptor.pixelFormat = .rgba8Unorm
accumulantTextureDescriptor.storageMode = .managed
accumulantTextureDescriptor.cpuCacheMode = .defaultCache
accumulantTextureDescriptor.resourceOptions = .storageModeManaged
let accumulantTexture = device.makeTexture(descriptor: accumulantTextureDescriptor)
/*
 End Pixel Color Buffer Setup
 */

commandEncoder.setComputePipelineState(combinePipeline)
commandEncoder.setBuffer(directionDataBuffer, offset: 0, index: 0)
commandEncoder.setTexture(accumulantTexture, index: 0)
commandEncoder.setBytes(&imageWidth, length: MemoryLayout<Int>.stride, index: 1)
commandEncoder.setBytes(&imageHeight, length: MemoryLayout<Int>.stride, index: 2)
commandEncoder.setBytes(&sampleCount, length: MemoryLayout<Int>.stride, index: 3)

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

if let blitEncoder = commandBuffer.makeBlitCommandEncoder() {
  blitEncoder.synchronize(resource: accumulantTexture!)
  blitEncoder.endEncoding()
}

let denoise = 

commandBuffer.commit()
commandBuffer.waitUntilCompleted()


if let error = commandBuffer.error as NSError? {
  if let encoderInfo = error.userInfo[MTLCommandBufferEncoderInfoErrorKey] as? [MTLCommandBufferEncoderInfo] {
    for info in encoderInfo {
      print(info.label + info.debugSignposts.joined())
    }
  }
}

saveImage(for: accumulantTexture!, where: "/Users/pprovins/Desktop/render.png")
