import Metal
import MetalPerformanceShaders
import Foundation
import simd

typealias Float4 = SIMD4<Float>
typealias Float3 = SIMD3<Float>

/// Converts a MTLTexture into a CGImage representation
/// - Parameter texture: The MTLTexture to convert into a PNG format CGImage
/// - Returns: Nil on failure or a CGImage that is in PNG format
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

/// Saves a MTLTexture as a PNG format image to a specified location
/// - Parameters:
///   - texture: the MTLTexture to save
///   - location: the absolute path where the image shall be saved
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

/// Given a compute pipeline and number of desired elements, return the most threads and threads per grid the kernel can be dispatched with
/// - Parameters:
///   - pipeline: The compute pipeline which will be dispatched
///   - count: The number of elements that can be operated on in parallel and independent of each other
/// - Returns: (MTLSize, MTLSize) -> (threadGroupsPerGrid, threadsPerThreadgroup)
func getOptimalThreadgroupSize(pipeline: MTLComputePipelineState, count: Int) -> (MTLSize, MTLSize) {
	let threadExecutionWidth = pipeline.maxTotalThreadsPerThreadgroup
	let threadgroupsPerGrid = MTLSize(width: (count + threadExecutionWidth - 1) / threadExecutionWidth, height: 1, depth: 1)
	let threadsPerThreadgroup = MTLSize(width: threadExecutionWidth, height: 1, depth: 1)
	
	return (threadgroupsPerGrid, threadsPerThreadgroup)
}

// MARK: Global Variables for Rendering
var imageWidth = 960
var imageHeight = 540
var pixelCount = UInt(imageWidth * imageHeight)
var sampleCount = 150
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

// MARK: Ray Spawning
/**
For each pixel, create an xyz origin and xyz direction.
Each pixel will have multiple samples, each buffer will hold samples * pixelCount float3 to accomodate for this.
The direction data buffer will also be reused to hold the pixel color for that sample so that another buffer does not need to be created.
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

var (threadgroupsPerGrid, threadsPerThreadgroup) = getOptimalThreadgroupSize(pipeline: spawnPipeline, count: Int(pixelCount))
commandEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)

// MARK: Ray Tracing Compute Launch
commandEncoder.setComputePipelineState(tracePipeline)
commandEncoder.setBuffer(originDataBuffer, offset: 0, index: 0)
commandEncoder.setBuffer(directionDataBuffer, offset: 0, index: 1)
commandEncoder.setBytes(&imageWidth, length: MemoryLayout<Int>.stride, index: 2)
commandEncoder.setBytes(&imageHeight, length: MemoryLayout<Int>.stride, index: 3)
commandEncoder.setBytes(&sampleCount, length: MemoryLayout<Int>.stride, index: 4)
commandEncoder.setBytes(&bounceCount, length: MemoryLayout<Int>.stride, index: 5)

(threadgroupsPerGrid, threadsPerThreadgroup) = getOptimalThreadgroupSize(pipeline: tracePipeline, count: Int(samplesByPixelsCount))
commandEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)

// MARK: Accumulation Into Final Texture
let accumulantTextureDescriptor = MTLTextureDescriptor()
accumulantTextureDescriptor.width = imageWidth
accumulantTextureDescriptor.height = imageHeight
accumulantTextureDescriptor.depth = 1
accumulantTextureDescriptor.usage = .shaderWrite
accumulantTextureDescriptor.mipmapLevelCount = 1
accumulantTextureDescriptor.allowGPUOptimizedContents = true
accumulantTextureDescriptor.pixelFormat = .rgba8Unorm
accumulantTextureDescriptor.storageMode = .managed
accumulantTextureDescriptor.cpuCacheMode = .defaultCache
accumulantTextureDescriptor.resourceOptions = .storageModeManaged
let accumulantTexture = device.makeTexture(descriptor: accumulantTextureDescriptor)

commandEncoder.setComputePipelineState(combinePipeline)
commandEncoder.setBuffer(directionDataBuffer, offset: 0, index: 0)
commandEncoder.setTexture(accumulantTexture, index: 0)
commandEncoder.setBytes(&imageWidth, length: MemoryLayout<Int>.stride, index: 1)
commandEncoder.setBytes(&imageHeight, length: MemoryLayout<Int>.stride, index: 2)
commandEncoder.setBytes(&sampleCount, length: MemoryLayout<Int>.stride, index: 3)

(threadgroupsPerGrid, threadsPerThreadgroup) = getOptimalThreadgroupSize(pipeline: combinePipeline, count: Int(pixelCount))
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

let homeDirectory = NSHomeDirectory()
let saveLocation = homeDirectory  + "/Desktop/render.png"
saveImage(for: accumulantTexture!, where: saveLocation)
