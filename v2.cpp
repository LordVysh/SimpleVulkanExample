#if defined(__ANDROID__)
#define VK_USE_PLATFORM_ANDROID_KHR
#elif defined(__linux__)
#define VK_USE_PLATFORM_XLIB_KHR
#elif defined(_WIN32)
#define VK_USE_PLATFORM_WIN32_KHR
#endif

#define THROW(exceptionClass, message) throw exceptionClass(__FILE__, __LINE__, (message) )

#define NOMINMAX
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define SDL_MAIN_HANDLED
#include <SDL2/SDL.h>
#include <SDL2/SDL_syswm.h>
#include <vulkan/vulkan.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include "stb-master/stb_image.h"

#include <iostream>
#include <set>
#include <chrono>
#include <algorithm>
#include <fstream>

const int windowHeight = 900;
const int windowWidth = 1800;

const std::vector<const char *> validationLayers = { "VK_LAYER_LUNARG_standard_validation" };
const std::vector<const char *> deviceExtensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

#ifdef _DEBUG
const bool enableValidationLayers = true;

#else
const bool enableValidationLayers = false;
#endif // _DEBUG

//Create Debug Report Callback Function Pointer
void vkCreateDebugReportCallbackEXT(vk::Instance& instance, vk::DebugReportCallbackCreateInfoEXT* pCreateInfo, vk::AllocationCallbacks* pAllocatorCallbacks, vk::DebugReportCallbackEXT* pCallback)
{
	auto func = (PFN_vkCreateDebugReportCallbackEXT)instance.getProcAddr("vkCreateDebugReportCallbackEXT");
	if (!func) throw std::runtime_error("Failed to get procedure address for vkCreateDebugReportCallbackEXT");
	func(*reinterpret_cast<VkInstance*>(&instance), reinterpret_cast<VkDebugReportCallbackCreateInfoEXT*>(pCreateInfo), reinterpret_cast<VkAllocationCallbacks*>(pAllocatorCallbacks), reinterpret_cast<VkDebugReportCallbackEXT*>(pCallback));
}

//Destroy Debug Report Callback Function Pointer
void vkDestroyDebugReportCallbackEXT(vk::Instance& instance, vk::DebugReportCallbackEXT* callback, vk::AllocationCallbacks* pAllocatorCallbacks)
{
	auto func = (PFN_vkDestroyDebugReportCallbackEXT)instance.getProcAddr("vkDestroyDebugReportCallbackEXT");
	if (!func) throw std::runtime_error("Failed to get procedure address for vkDestroyDebugReportCallbackEXT");
	func(*reinterpret_cast<VkInstance*>(&instance), *reinterpret_cast<VkDebugReportCallbackEXT*>(callback), reinterpret_cast<VkAllocationCallbacks*>(pAllocatorCallbacks));
}

//Generic Debug Message Handler
VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
	VkDebugReportFlagsEXT flags,
	VkDebugReportObjectTypeEXT objType,
	uint64_t obj,
	size_t location,
	int32_t code,
	const char* layerPrefix,
	const char* msg,
	void* userData
)
{
	std::cerr << "[Layer: "
		<< layerPrefix
		<< "]\n[Message: "
		<< msg
		<< "]\n";
	return VK_FALSE;
}

//Creates a Vulkan surface
vk::SurfaceKHR createVulkanSurface(const vk::Instance& instance, SDL_Window* window)
{
	SDL_SysWMinfo windowInfo;
	SDL_VERSION(&windowInfo.version);
	if (!SDL_GetWindowWMInfo(window, &windowInfo)) {
		throw std::system_error(std::error_code(), "SDK window manager info is not available.");
	}

	switch (windowInfo.subsystem) {

#if defined(SDL_VIDEO_DRIVER_ANDROID) && defined(VK_USE_PLATFORM_ANDROID_KHR)
	case SDL_SYSWM_ANDROID: {
		vk::AndroidSurfaceCreateInfoKHR surfaceInfo = vk::AndroidSurfaceCreateInfoKHR()
			.setWindow(windowInfo.info.android.window);
		return instance.createAndroidSurfaceKHR(surfaceInfo);
	}
#endif

#if defined(SDL_VIDEO_DRIVER_MIR) && defined(VK_USE_PLATFORM_MIR_KHR)
	case SDL_SYSWM_MIR: {
		vk::MirSurfaceCreateInfoKHR surfaceInfo = vk::MirSurfaceCreateInfoKHR()
			.setConnection(windowInfo.info.mir.connection)
			.setMirSurface(windowInfo.info.mir.surface);
		return instance.createMirSurfaceKHR(surfaceInfo);
	}
#endif

#if defined(SDL_VIDEO_DRIVER_WAYLAND) && defined(VK_USE_PLATFORM_WAYLAND_KHR)
	case SDL_SYSWM_WAYLAND: {
		vk::WaylandSurfaceCreateInfoKHR surfaceInfo = vk::WaylandSurfaceCreateInfoKHR()
			.setDisplay(windowInfo.info.wl.display)
			.setSurface(windowInfo.info.wl.surface);
		return instance.createWaylandSurfaceKHR(surfaceInfo);
	}
#endif

#if defined(SDL_VIDEO_DRIVER_WINDOWS) && defined(VK_USE_PLATFORM_WIN32_KHR)
	case SDL_SYSWM_WINDOWS: {
		vk::Win32SurfaceCreateInfoKHR surfaceInfo = vk::Win32SurfaceCreateInfoKHR()
			.setHinstance(GetModuleHandle(NULL))
			.setHwnd(windowInfo.info.win.window);
		return instance.createWin32SurfaceKHR(surfaceInfo);
	}
#endif

#if defined(SDL_VIDEO_DRIVER_X11) && defined(VK_USE_PLATFORM_XLIB_KHR)
	case SDL_SYSWM_X11: {
		vk::XlibSurfaceCreateInfoKHR surfaceInfo = vk::XlibSurfaceCreateInfoKHR()
			.setDpy(windowInfo.info.x11.display)
			.setWindow(windowInfo.info.x11.window);
		return instance.createXlibSurfaceKHR(surfaceInfo);
	}
#endif

	default:
		throw std::system_error(std::error_code(), "Unsupported window manager is in use.");
	}
}

std::vector<const char*> getAvailableWSIExtensions()
{
	std::vector<const char*> extensions;
	extensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);

#if defined(VK_USE_PLATFORM_ANDROID_KHR)
	extensions.push_back(VK_KHR_ANDROID_SURFACE_EXTENSION_NAME);
#endif
#if defined(VK_USE_PLATFORM_MIR_KHR)
	extensions.push_back(VK_KHR_MIR_SURFACE_EXTENSION_NAME);
#endif
#if defined(VK_USE_PLATFORM_WAYLAND_KHR)
	extensions.push_back(VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME);
#endif
#if defined(VK_USE_PLATFORM_WIN32_KHR)
	extensions.push_back(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
#endif
#if defined(VK_USE_PLATFORM_XLIB_KHR)
	extensions.push_back(VK_KHR_XLIB_SURFACE_EXTENSION_NAME);
#endif

	return extensions;
}

//Store our queue family indices
struct queueFamilyIndices
{
	int graphicsIndex = -1;
	int presentIndex = -1;

	bool isComplete()
	{
		return (graphicsIndex >= 0 && presentIndex >= 0);
	}
};

//Store information needed for swapchain
struct swapchainDetails
{
	vk::SurfaceCapabilitiesKHR capabilities;
	std::vector<vk::SurfaceFormatKHR> formats;
	std::vector<vk::PresentModeKHR> presentModes;
};

//Create the vertex information for the triangle to render
struct Vertex
{
	glm::vec3 position;
	glm::vec3 color;
	glm::vec2 textureCoordinate;

	static vk::VertexInputBindingDescription getBindingDescription()
	{
		vk::VertexInputBindingDescription bindingDescription = vk::VertexInputBindingDescription()
			.setBinding(0)
			.setStride(sizeof(Vertex))
			.setInputRate(vk::VertexInputRate::eVertex);

		return bindingDescription;
	}

	static std::vector<vk::VertexInputAttributeDescription> getAttributeDescription()
	{
		std::vector<vk::VertexInputAttributeDescription> attributeDescriptions(3);

		attributeDescriptions[0].setBinding(0)
			.setLocation(0)
			.setFormat(vk::Format::eR32G32B32Sfloat)
			.setOffset(offsetof(Vertex, position));

		attributeDescriptions[1].setBinding(0)
			.setLocation(1)
			.setFormat(vk::Format::eR32G32B32Sfloat)
			.setOffset(offsetof(Vertex, color));

		attributeDescriptions[2].setBinding(0)
			.setLocation(2)
			.setFormat(vk::Format::eR32G32B32Sfloat)
			.setOffset(offsetof(Vertex, textureCoordinate));

		return attributeDescriptions;
	}
};

struct uniformBufferObject
{
	glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
};

const std::vector<Vertex> vertices = {
	{ { -0.5f, -0.5f, 0.0f },{ 1.0f, 0.0f, 0.0f },{ 0.0f, 0.0f } },
	{ { 0.5f, -0.5f, 0.0f },{ 0.0f, 1.0f, 0.0f },{ 1.0f, 0.0f } },
	{ { 0.5f, 0.5f, 0.0f },{ 0.0f, 0.0f, 1.0f },{ 1.0f, 1.0f } },
	{ { -0.5f, 0.5f, 0.0f },{ 1.0f, 1.0f, 1.0f },{ 0.0f, 1.0f } },

	{ { -0.5f, -0.5f, -0.5f },{ 1.0f, 0.0f, 0.0f },{ 0.0f, 0.0f } },
	{ { 0.5f, -0.5f, -0.5f },{ 0.0f, 1.0f, 0.0f },{ 1.0f, 0.0f } },
	{ { 0.5f, 0.5f, -0.5f },{ 0.0f, 0.0f, 1.0f },{ 1.0f, 1.0f } },
	{ { -0.5f, 0.5f, -0.5f },{ 1.0f, 1.0f, 1.0f },{ 0.0f, 1.0f } }
};

const std::vector<uint16_t> indices = {
	0, 1, 2, 2, 3, 0,
	4, 5, 6, 6, 7, 4
};

class VulkanTestV2
{
public:
	void run()
	{
		initWindow();
		initVulkan();
		mainLoop();
		cleanup();
	}

private:
	//SDL
	SDL_Window* window;

	//Instance
	vk::Instance instance;
	vk::DebugReportCallbackEXT debugCallbackEXT;
	vk::SurfaceKHR surface;

	//Physical and Logical Device
	vk::PhysicalDevice physicalDevice = VK_NULL_HANDLE;
	vk::Device device;

	//Queues
	vk::Queue graphicsQueue;
	vk::Queue presentQueue;

	//Swapchain
	vk::SwapchainKHR swapchain;
	std::vector<vk::Image> swapchainImages;
	vk::Format swapchainImageFormat;
	vk::Extent2D swapchainExtent;
	std::vector<vk::ImageView> swapchainImageViews;
	std::vector<vk::Framebuffer> swapchainFramebuffers;

	//Render Pass and Pipeline
	vk::RenderPass renderPass;
	vk::DescriptorSetLayout descriptorSetLayout;
	vk::PipelineLayout pipelineLayout;
	vk::Pipeline graphicsPipeline;

	//Command Pool
	vk::CommandPool commandPool;

	//Depth Buffer
	vk::Image depthImage;
	vk::DeviceMemory depthMemory;
	vk::ImageView depthImageView;

	//Texture
	vk::Image textureImage;
	vk::DeviceMemory textureMemory;
	vk::ImageView textureImageView;
	vk::Sampler textureSampler;

	//Vertex/Index Buffer
	vk::Buffer vertexBuffer;
	vk::DeviceMemory vertexMemory;
	vk::Buffer indexBuffer;
	vk::DeviceMemory indexMemory;

	//Uniform Buffer
	vk::Buffer uniformBuffer;
	vk::DeviceMemory uniformMemory;

	//Descriptor Pool
	vk::DescriptorPool descriptorPool;
	vk::DescriptorSet descriptorSet;

	//Command Buffers
	std::vector<vk::CommandBuffer> commandBuffers;

	//Semaphores
	vk::Semaphore imageAvailableSemaphore;
	vk::Semaphore renderFinishedSemaphore;

	void initWindow()
	{
		SDL_Init(SDL_INIT_VIDEO);

		window = SDL_CreateWindow("Vulkan Window", SDL_WINDOWPOS_CENTERED,
			SDL_WINDOWPOS_CENTERED, windowWidth, windowHeight, SDL_WINDOW_OPENGL);

	}

	void initVulkan() {
		createInstance();
		setupDebugCallback();
		createSurface();
		pickPhysicalDevice();
		createLogicalDevice();
		createSwapchain();
		createImageViews();
		createRenderPass();
		createDescriptorSetLayout();
		createGraphicsPipeline();
		createCommandPool();
		createDepthResources();
		createFramebuffers();
		createTextureImage();
		createTextureImageView();
		createTextureSampler();
		createVertexBuffer();
		createIndexBuffer();
		createUniformBuffer();
		createDescriptorPool();
		createDescriptorSet();
		createCommandBuffers();
		createSemaphores();
	}

	void mainLoop() {
		bool stillRunning = true;
		while (stillRunning) {

			SDL_Event event;
			while (SDL_PollEvent(&event)) {

				switch (event.type) {

				case SDL_QUIT:
					stillRunning = false;
					break;

				default:
					// Do nothing.
					break;
				}
			}
			updateUniformBuffer();
			drawFrame();
			SDL_Delay(10);
		}

			device.waitIdle();
	}

	void cleanupSwapchain()
	{
		device.destroyImageView(depthImageView);
		device.destroyImage(depthImage);
		device.freeMemory(depthMemory);

		for (size_t i = 0; i < swapchainFramebuffers.size(); i++)
		{
			device.destroyFramebuffer(swapchainFramebuffers[i]);
		}

		device.freeCommandBuffers(commandPool, commandBuffers);

		device.destroyPipeline(graphicsPipeline);
		device.destroyPipelineLayout(pipelineLayout);
		device.destroyRenderPass(renderPass);

		for (size_t i = 0; i < swapchainImageViews.size(); i++)
		{
			device.destroyImageView(swapchainImageViews[i]);
		}

		device.destroySwapchainKHR(swapchain);
	}

	void cleanup()
	{
		cleanupSwapchain();

		device.destroySampler(textureSampler);
		device.destroyImageView(textureImageView);

		device.destroyImage(textureImage);
		device.freeMemory(textureMemory);

		device.destroyDescriptorPool(descriptorPool);

		device.destroyDescriptorSetLayout(descriptorSetLayout);
		device.destroyBuffer(uniformBuffer);
		device.freeMemory(uniformMemory);

		device.destroyBuffer(indexBuffer);
		device.freeMemory(indexMemory);

		device.destroyBuffer(vertexBuffer);
		device.freeMemory(vertexMemory);

		device.destroySemaphore(renderFinishedSemaphore);
		device.destroySemaphore(imageAvailableSemaphore);

		device.destroyCommandPool(commandPool);

		device.waitIdle();
		device.destroy();
		vkDestroyDebugReportCallbackEXT(instance, &debugCallbackEXT, nullptr);
		instance.destroySurfaceKHR(surface);
		instance.destroy();

		SDL_DestroyWindow(window);
		SDL_Quit();

	}

	static void onWindowResized(VulkanTestV2 * app, SDL_Window* window, int width, int height)
	{
		if (width == 0 || height == 0) return;

		app->recreateSwapchain();
	}

	void recreateSwapchain()
	{
		device.waitIdle();
		cleanupSwapchain();

		createSwapchain();
		createImageViews();
		createRenderPass();
		createGraphicsPipeline();
		createDepthResources();
		createFramebuffers();
		createCommandBuffers();
	}

	void createInstance()
	{
		if (enableValidationLayers && !checkValidationLayerSupport())
		{
			throw std::runtime_error("Validation layers requested, but not availabel!");
		}

		vk::ApplicationInfo appInfo = vk::ApplicationInfo()
			.setPApplicationName("Vulkan Test v2.0")
			.setApplicationVersion(2)
			.setPEngineName("GlassBox")
			.setEngineVersion(1)
			.setApiVersion(VK_API_VERSION_1_0);

		vk::InstanceCreateInfo createInfo = vk::InstanceCreateInfo()
			.setPApplicationInfo(&appInfo);

		auto extensions = getRequiredExtensions();
		createInfo.setEnabledExtensionCount(static_cast<uint32_t>(extensions.size()))
			.setPpEnabledExtensionNames(extensions.data());

		if (enableValidationLayers)
		{
			createInfo.setEnabledLayerCount(static_cast<uint32_t>(validationLayers.size()))
				.setPpEnabledLayerNames(validationLayers.data());
		}
		else
		{
			createInfo.setEnabledLayerCount(0);
		}
		try {
			instance = vk::createInstance(createInfo);
		}
		catch (const std::exception& e) {
			std::cout << "Could not create a Vulkan instance: " << e.what() << std::endl;
		}
	}

	void setupDebugCallback()
	{
		if (!enableValidationLayers) return;
		vk::DebugReportCallbackCreateInfoEXT createInfo = vk::DebugReportCallbackCreateInfoEXT()
			.setFlags(vk::DebugReportFlagBitsEXT::eError | vk::DebugReportFlagBitsEXT::eWarning | vk::DebugReportFlagBitsEXT::eInformation)
			.setPfnCallback(debugCallback);
		vkCreateDebugReportCallbackEXT(instance, &createInfo, NULL, &debugCallbackEXT);
	}

	void createSurface()
	{
		try {
			surface = createVulkanSurface(instance, window);
		}
		catch (const std::exception& e) {
			std::cout << "Failed to create Vulkan surface: " << e.what() << std::endl;
			instance.destroy();
		}
	}

	void pickPhysicalDevice()
	{
		std::vector<vk::PhysicalDevice> physicalDevices = instance.enumeratePhysicalDevices();

		if (physicalDevices.size() == 0)
		{
			throw std::runtime_error("Failed to find device with Vulkan support!");
		}

		for (const auto& device : physicalDevices)
		{
			if (isDeviceSuitable(device))
			{
				physicalDevice = device;
				break;
			}
		}

		if (physicalDevice == VK_NULL_HANDLE)
		{
			throw std::runtime_error("No suitable devices found!");
		}
	}

	void createLogicalDevice()
	{
		queueFamilyIndices indices = findQueueFamilies(physicalDevice);

		std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
		std::set<int> uniqueQueueFamilies = { indices.graphicsIndex, indices.presentIndex };

		float queuePriority = 1.0f;
		for (int queueFamily : uniqueQueueFamilies)
		{
			vk::DeviceQueueCreateInfo queueCreateInfo = vk::DeviceQueueCreateInfo()
				.setQueueFamilyIndex(queueFamily)
				.setQueueCount(1)
				.setPQueuePriorities(&queuePriority);
			queueCreateInfos.push_back(queueCreateInfo);
		}

		vk::PhysicalDeviceFeatures deviceFeatures = vk::PhysicalDeviceFeatures()
			.setSamplerAnisotropy(VK_TRUE)
			.setDepthClamp(VK_TRUE);

		vk::DeviceCreateInfo createInfo = vk::DeviceCreateInfo()
			.setQueueCreateInfoCount(static_cast<uint32_t>(queueCreateInfos.size()))
			.setPQueueCreateInfos(queueCreateInfos.data())
			.setPEnabledFeatures(&deviceFeatures)
			.setEnabledExtensionCount(static_cast<uint32_t>(deviceExtensions.size()))
			.setPpEnabledExtensionNames(deviceExtensions.data());

		if (enableValidationLayers)
		{
			createInfo.setEnabledLayerCount(static_cast<uint32_t>(validationLayers.size()))
				.setPpEnabledLayerNames(validationLayers.data());
		}
		else
		{
			createInfo.setEnabledLayerCount(0);
		}
		device = physicalDevice.createDevice(createInfo);

		graphicsQueue = device.getQueue(indices.graphicsIndex, 0);
		presentQueue = device.getQueue(indices.presentIndex, 0);
	}

	void createSwapchain()
	{
		swapchainDetails swapchainSupport = querySwapchainSupport(physicalDevice);

		vk::SurfaceFormatKHR surfaceFormat = chooseSwapchainSurfaceFormat(swapchainSupport.formats);
		vk::PresentModeKHR presentMode = chooseSwapchainPresentMode(swapchainSupport.presentModes);
		vk::Extent2D extent = chooseSwapchainExtent(swapchainSupport.capabilities);

		uint32_t imageCount = swapchainSupport.capabilities.minImageCount + 1;
		if (swapchainSupport.capabilities.maxImageCount > 0 && imageCount > swapchainSupport.capabilities.maxImageCount)
		{
			imageCount = swapchainSupport.capabilities.maxImageCount;
		}

		vk::SwapchainCreateInfoKHR createInfo = vk::SwapchainCreateInfoKHR()
			.setSurface(surface)
			.setMinImageCount(imageCount)
			.setImageFormat(surfaceFormat.format)
			.setImageColorSpace(surfaceFormat.colorSpace)
			.setImageExtent(extent)
			.setImageArrayLayers(1)
			.setImageUsage(vk::ImageUsageFlagBits::eColorAttachment);

		queueFamilyIndices indices = findQueueFamilies(physicalDevice);
		uint32_t queueFamilyIndicesARR[] = { (uint32_t)indices.graphicsIndex, (uint32_t)indices.presentIndex };
		
		if (indices.graphicsIndex != indices.presentIndex)
		{
			createInfo.setImageSharingMode(vk::SharingMode::eConcurrent)
				.setQueueFamilyIndexCount(2)
				.setPQueueFamilyIndices(queueFamilyIndicesARR);
		}
		else
		{
			createInfo.setImageSharingMode(vk::SharingMode::eExclusive);
		}

		createInfo.setPreTransform(swapchainSupport.capabilities.currentTransform)
			.setCompositeAlpha(vk::CompositeAlphaFlagBitsKHR::eOpaque)
			.setPresentMode(presentMode)
			.setClipped(VK_TRUE);

		swapchain = device.createSwapchainKHR(createInfo);

		swapchainImages = device.getSwapchainImagesKHR(swapchain);
		swapchainImageFormat = surfaceFormat.format;
		swapchainExtent = extent;
	}

	void createImageViews()
	{
		swapchainImageViews.resize(swapchainImages.size());

		for (size_t i = 0; i < swapchainImages.size(); i++)
		{
			swapchainImageViews[i] = createImageView(swapchainImages[i], swapchainImageFormat, vk::ImageAspectFlagBits::eColor);
		}
	}

	void createRenderPass()
	{
		vk::AttachmentDescription colorAttachment = vk::AttachmentDescription()
			.setFormat(swapchainImageFormat)
			.setSamples(vk::SampleCountFlagBits::e1)
			.setLoadOp(vk::AttachmentLoadOp::eClear)
			.setStoreOp(vk::AttachmentStoreOp::eStore)
			.setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
			.setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
			.setInitialLayout(vk::ImageLayout::eUndefined)
			.setFinalLayout(vk::ImageLayout::ePresentSrcKHR);

		vk::AttachmentDescription depthAttachment = vk::AttachmentDescription()
			.setFormat(vk::Format::eD16Unorm)
			.setSamples(vk::SampleCountFlagBits::e1)
			.setLoadOp(vk::AttachmentLoadOp::eClear)
			.setStoreOp(vk::AttachmentStoreOp::eDontCare)
			.setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
			.setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
			.setInitialLayout(vk::ImageLayout::eUndefined)
			.setFinalLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal);

		vk::AttachmentReference colorAttachmentRef = vk::AttachmentReference()
			.setAttachment(0)
			.setLayout(vk::ImageLayout::eColorAttachmentOptimal);

		vk::AttachmentReference depthAttachmentRef = vk::AttachmentReference()
			.setAttachment(1)
			.setLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal);

		vk::SubpassDescription subpass = vk::SubpassDescription()
			.setPipelineBindPoint(vk::PipelineBindPoint::eGraphics)
			.setColorAttachmentCount(1)
			.setPColorAttachments(&colorAttachmentRef)
			.setPDepthStencilAttachment(&depthAttachmentRef);

		vk::SubpassDependency dependency = vk::SubpassDependency()
			.setSrcSubpass(VK_SUBPASS_EXTERNAL)
			.setDstSubpass(0)
			.setSrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput)
			.setDstStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput)
			.setDstAccessMask(vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite);

		std::vector<vk::AttachmentDescription> attachments = { colorAttachment, depthAttachment };
		vk::RenderPassCreateInfo renderPassInfo = vk::RenderPassCreateInfo()
			.setAttachmentCount(static_cast<uint32_t>(attachments.size()))
			.setPAttachments(attachments.data())
			.setSubpassCount(1)
			.setPSubpasses(&subpass)
			.setDependencyCount(1)
			.setPDependencies(&dependency);

		renderPass = device.createRenderPass(renderPassInfo);
	}

	void createDescriptorSetLayout()
	{
		vk::DescriptorSetLayoutBinding uniformBufferLayoutBinding = vk::DescriptorSetLayoutBinding()
			.setBinding(0)
			.setDescriptorCount(1)
			.setDescriptorType(vk::DescriptorType::eUniformBuffer)
			.setPImmutableSamplers(nullptr)
			.setStageFlags(vk::ShaderStageFlagBits::eVertex);

		vk::DescriptorSetLayoutBinding samplerLayoutBinding = vk::DescriptorSetLayoutBinding()
			.setBinding(1)
			.setDescriptorCount(1)
			.setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
			.setPImmutableSamplers(nullptr)
			.setStageFlags(vk::ShaderStageFlagBits::eFragment);

		std::vector<vk::DescriptorSetLayoutBinding> bindings = { uniformBufferLayoutBinding, samplerLayoutBinding };
		vk::DescriptorSetLayoutCreateInfo layoutInfo = vk::DescriptorSetLayoutCreateInfo()
			.setBindingCount(static_cast<uint32_t>(bindings.size()))
			.setPBindings(bindings.data());

		descriptorSetLayout = device.createDescriptorSetLayout(layoutInfo);
	}

	void createGraphicsPipeline()
	{
		std::vector<char> vertShaderCode;
		std::vector<char> fragShaderCode;
		try {
			vertShaderCode = readFile("\\shaders\\vert.spv");
			fragShaderCode = readFile("\\shaders\\frag.spv");
		}
		catch (std::exception &e)
		{
			std::cerr << e.what() << '\n';
		}

		vk::ShaderModule vertShaderModule = createShaderModule(vertShaderCode);
		vk::ShaderModule fragShaderModule = createShaderModule(fragShaderCode);

		vk::PipelineShaderStageCreateInfo vertShaderStageInfo = vk::PipelineShaderStageCreateInfo()
			.setStage(vk::ShaderStageFlagBits::eVertex)
			.setModule(vertShaderModule)
			.setPName("main");

		vk::PipelineShaderStageCreateInfo fragShaderStageInfo = vk::PipelineShaderStageCreateInfo()
			.setStage(vk::ShaderStageFlagBits::eFragment)
			.setModule(fragShaderModule)
			.setPName("main");

		vk::PipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

		vk::PipelineVertexInputStateCreateInfo vertexInputInfo;

		auto bindingDescription = Vertex::getBindingDescription();
		auto attributeDescriptions = Vertex::getAttributeDescription();

		vertexInputInfo.setVertexBindingDescriptionCount(1)
			.setVertexAttributeDescriptionCount(static_cast<uint32_t>(attributeDescriptions.size()))
			.setPVertexBindingDescriptions(&bindingDescription)
			.setPVertexAttributeDescriptions(attributeDescriptions.data());

		vk::PipelineInputAssemblyStateCreateInfo inputAssembly = vk::PipelineInputAssemblyStateCreateInfo()
			.setTopology(vk::PrimitiveTopology::eTriangleList)
			.setPrimitiveRestartEnable(VK_FALSE);

		vk::Viewport viewport = vk::Viewport()
			.setX(0.0f)
			.setY(0.0f)
			.setWidth((float)swapchainExtent.width)
			.setHeight((float)swapchainExtent.height)
			.setMinDepth(0.0f)
			.setMaxDepth(1.0f);

		vk::Rect2D scissor = vk::Rect2D()
			.setOffset({ 0, 0 })
			.setExtent(swapchainExtent);

		vk::PipelineViewportStateCreateInfo viewportState = vk::PipelineViewportStateCreateInfo()
			.setViewportCount(1)
			.setPViewports(&viewport)
			.setScissorCount(1)
			.setPScissors(&scissor);

		vk::PipelineRasterizationStateCreateInfo rasterizer = vk::PipelineRasterizationStateCreateInfo()
			.setDepthClampEnable(VK_FALSE)
			.setRasterizerDiscardEnable(VK_FALSE)
			.setPolygonMode(vk::PolygonMode::eFill)
			.setLineWidth(1.0f)
			.setCullMode(vk::CullModeFlagBits::eBack)
			.setFrontFace(vk::FrontFace::eClockwise)
			.setDepthBiasEnable(VK_FALSE);

		vk::PipelineMultisampleStateCreateInfo multisampling = vk::PipelineMultisampleStateCreateInfo()
			.setSampleShadingEnable(VK_FALSE)
			.setRasterizationSamples(vk::SampleCountFlagBits::e1);

		vk::PipelineDepthStencilStateCreateInfo depthStencil = vk::PipelineDepthStencilStateCreateInfo()
			.setDepthTestEnable(VK_TRUE)
			.setDepthWriteEnable(VK_TRUE)
			.setDepthCompareOp(vk::CompareOp::eLess)
			.setDepthBoundsTestEnable(VK_FALSE)
			.setStencilTestEnable(VK_FALSE);

		vk::PipelineColorBlendAttachmentState colorBlendAttachment = vk::PipelineColorBlendAttachmentState()
			.setColorWriteMask(vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eA)
			.setBlendEnable(VK_FALSE);

		vk::PipelineColorBlendStateCreateInfo colorBlending = vk::PipelineColorBlendStateCreateInfo()
			.setLogicOpEnable(VK_FALSE)
			.setLogicOp(vk::LogicOp::eCopy)
			.setAttachmentCount(1)
			.setPAttachments(&colorBlendAttachment)
			.setBlendConstants({ 0.0f, 0.0f, 0.0f, 0.0f });

		vk::PipelineLayoutCreateInfo pipelineLayoutInfo = vk::PipelineLayoutCreateInfo()
			.setSetLayoutCount(1)
			.setPSetLayouts(&descriptorSetLayout);

		pipelineLayout = device.createPipelineLayout(pipelineLayoutInfo);

		vk::GraphicsPipelineCreateInfo pipelineInfo = vk::GraphicsPipelineCreateInfo()
			.setStageCount(2)
			.setPStages(shaderStages)
			.setPVertexInputState(&vertexInputInfo)
			.setPInputAssemblyState(&inputAssembly)
			.setPViewportState(&viewportState)
			.setPRasterizationState(&rasterizer)
			.setPMultisampleState(&multisampling)
			.setPDepthStencilState(&depthStencil)
			.setPColorBlendState(&colorBlending)
			.setLayout(pipelineLayout)
			.setRenderPass(renderPass)
			.setSubpass(0)
			.setBasePipelineHandle(VK_NULL_HANDLE);

		graphicsPipeline = device.createGraphicsPipeline(VK_NULL_HANDLE, pipelineInfo);

		device.destroyShaderModule(vertShaderModule);
		device.destroyShaderModule(fragShaderModule);
	}

	void createFramebuffers()
	{
		swapchainFramebuffers.resize(swapchainImageViews.size());

		for (size_t i = 0; i < swapchainImageViews.size(); i++)
		{
			std::vector<vk::ImageView> attachments = 
			{
				swapchainImageViews[i],
				depthImageView
			};

			vk::FramebufferCreateInfo frameBufferInfo = vk::FramebufferCreateInfo()
				.setRenderPass(renderPass)
				.setAttachmentCount(static_cast<uint32_t>(attachments.size()))
				.setPAttachments(attachments.data())
				.setWidth(swapchainExtent.width)
				.setHeight(swapchainExtent.height)
				.setLayers(1);

			swapchainFramebuffers[i] = device.createFramebuffer(frameBufferInfo);
		}

	}

	void createCommandPool()
	{
		queueFamilyIndices indices = findQueueFamilies(physicalDevice);

		vk::CommandPoolCreateInfo poolInfo = vk::CommandPoolCreateInfo()
			.setQueueFamilyIndex(indices.graphicsIndex);

		commandPool = device.createCommandPool(poolInfo);
	}

	void createDepthResources()
	{
		vk::Format depthFormat = vk::Format::eD16Unorm;

		createImage(swapchainExtent.width, swapchainExtent.height, depthFormat, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::MemoryPropertyFlagBits::eDeviceLocal, depthImage, depthMemory);
		depthImageView = createImageView(depthImage, depthFormat, vk::ImageAspectFlagBits::eDepth);

		transitionImageLayout(depthImage, depthFormat, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal);
	}

	vk::Format findSupportedFormat(const std::vector<vk::Format>& candidates, vk::ImageTiling tiling, vk::FormatFeatureFlags features)
	{
		for (vk::Format format : candidates)
		{
			vk::FormatProperties properties = physicalDevice.getFormatProperties(format);

			if (tiling == vk::ImageTiling::eLinear && (properties.linearTilingFeatures & features) == features)
			{
				return format;
			}
			else if (tiling == vk::ImageTiling::eOptimal && (properties.optimalTilingFeatures & features) == features)
			{
				return format;
			}
		}

		throw std::runtime_error("Failed to find supported format!");
	}

	vk::Format findDepthFormat()
	{
		return findSupportedFormat(
		{ vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint },
			vk::ImageTiling::eOptimal,
			vk::FormatFeatureFlagBits::eDepthStencilAttachment
		);
	}

	bool hasStencilComponent(vk::Format format)
	{
		return format == vk::Format::eD32SfloatS8Uint || format == vk::Format::eD24UnormS8Uint;
	}

	void createTextureImage()
	{
		int texWidth, texHeight, texChannels;
		stbi_uc* pixels = stbi_load("textures/texture.jpg", &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
		vk::DeviceSize imageSize = texWidth * texHeight * 4;

		if (!pixels)
		{
			throw std::runtime_error("Unable to load image!");
		}

		vk::Buffer stagingBuffer;
		vk::DeviceMemory stagingBufferMemory;
		createBuffer(imageSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

		void* data;
		device.mapMemory(stagingBufferMemory, vk::DeviceSize(0), imageSize, vk::MemoryMapFlagBits(0), &data);
		memcpy(data, pixels, static_cast<size_t>(imageSize));
		device.unmapMemory(stagingBufferMemory);

		stbi_image_free(pixels);

		createImage(texWidth, texHeight, vk::Format::eR8G8B8A8Unorm, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled, vk::MemoryPropertyFlagBits::eDeviceLocal, textureImage, textureMemory);

		transitionImageLayout(textureImage, vk::Format::eR8G8B8A8Unorm, vk::ImageLayout::ePreinitialized, vk::ImageLayout::eTransferDstOptimal);
		copyBufferToImage(stagingBuffer, textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
		transitionImageLayout(textureImage, vk::Format::eR8G8B8A8Unorm, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);

		device.destroyBuffer(stagingBuffer);
		device.freeMemory(stagingBufferMemory);
	}

	void createTextureImageView()
	{
		textureImageView = createImageView(textureImage, vk::Format::eR8G8B8A8Unorm, vk::ImageAspectFlagBits::eColor);
	}

	void createTextureSampler()
	{
		vk::SamplerCreateInfo samplerInfo = vk::SamplerCreateInfo()
			.setMagFilter(vk::Filter::eLinear)
			.setMinFilter(vk::Filter::eLinear)
			.setAddressModeU(vk::SamplerAddressMode::eRepeat)
			.setAddressModeV(vk::SamplerAddressMode::eRepeat)
			.setAddressModeW(vk::SamplerAddressMode::eRepeat)
			.setAnisotropyEnable(VK_TRUE)
			.setMaxAnisotropy(16)
			.setBorderColor(vk::BorderColor::eIntOpaqueBlack)
			.setUnnormalizedCoordinates(VK_FALSE)
			.setCompareEnable(VK_FALSE)
			.setCompareOp(vk::CompareOp::eAlways)
			.setMipmapMode(vk::SamplerMipmapMode::eLinear);

		textureSampler = device.createSampler(samplerInfo);
	}

	vk::ImageView createImageView(vk::Image image, vk::Format format, vk::ImageAspectFlags aspectFlags)
	{
		vk::ImageSubresourceRange subresourceRange = vk::ImageSubresourceRange()
			.setAspectMask(aspectFlags)
			.setBaseMipLevel(0)
			.setLevelCount(1)
			.setBaseArrayLayer(0)
			.setLayerCount(1);
		vk::ImageViewCreateInfo viewInfo = vk::ImageViewCreateInfo()
			.setImage(image)
			.setViewType(vk::ImageViewType::e2D)
			.setFormat(format)
			.setSubresourceRange(subresourceRange);

		vk::ImageView imageView = device.createImageView(viewInfo);

		return imageView;
	}

	void createImage(uint32_t width, uint32_t height, vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties, vk::Image& image, vk::DeviceMemory& memory)
	{
		vk::ImageCreateInfo imageInfo = vk::ImageCreateInfo()
			.setImageType(vk::ImageType::e2D)
			.setExtent(vk::Extent3D(width, height, 1))
			.setMipLevels(1)
			.setArrayLayers(1)
			.setFormat(format)
			.setTiling(tiling)
			.setInitialLayout(vk::ImageLayout::ePreinitialized)
			.setUsage(usage)
			.setSamples(vk::SampleCountFlagBits::e1)
			.setSharingMode(vk::SharingMode::eExclusive);

		image = device.createImage(imageInfo);

		vk::MemoryRequirements memRequirements = device.getImageMemoryRequirements(image);

		vk::MemoryAllocateInfo allocateInfo = vk::MemoryAllocateInfo()
			.setAllocationSize(memRequirements.size)
			.setMemoryTypeIndex(findMemoryType(memRequirements.memoryTypeBits, properties));

		memory = device.allocateMemory(allocateInfo);

		device.bindImageMemory(image, memory, 0);
	}

	void transitionImageLayout(vk::Image image, vk::Format format, vk::ImageLayout oldLayout, vk::ImageLayout newLayout)
	{
		vk::CommandBuffer commandBuffer = beginSingleTimeCommands();

		vk::ImageSubresourceRange subresourceRange = vk::ImageSubresourceRange()
			.setBaseMipLevel(0)
			.setLevelCount(1)
			.setBaseArrayLayer(0)
			.setLayerCount(1);

		vk::ImageMemoryBarrier barrier = vk::ImageMemoryBarrier()
			.setOldLayout(oldLayout)
			.setNewLayout(newLayout)
			.setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
			.setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
			.setImage(image);

		if (newLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal)
		{
			subresourceRange.setAspectMask(vk::ImageAspectFlagBits::eDepth);
			if (hasStencilComponent(format))
			{
				subresourceRange.setAspectMask(vk::ImageAspectFlagBits::eDepth);
			}
		}
		else
		{
			subresourceRange.setAspectMask(vk::ImageAspectFlagBits::eColor);
		}

		barrier.setSubresourceRange(subresourceRange);

		if (oldLayout == vk::ImageLayout::ePreinitialized && newLayout == vk::ImageLayout::eTransferDstOptimal)
		{
			barrier.setSrcAccessMask(vk::AccessFlagBits::eHostWrite)
				.setDstAccessMask(vk::AccessFlagBits::eTransferWrite);
		}
		else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal)
		{
			barrier.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
				.setDstAccessMask(vk::AccessFlagBits::eShaderRead);
		}
		else if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal)
		{
			barrier.setSrcAccessMask(vk::AccessFlagBits(0))
				.setDstAccessMask(vk::AccessFlagBits::eDepthStencilAttachmentRead | vk::AccessFlagBits::eDepthStencilAttachmentWrite);
		}
		else
		{
			throw std::runtime_error("Unsupported layout transition!");
		}

		commandBuffer.pipelineBarrier(
			vk::PipelineStageFlagBits::eTopOfPipe,
			vk::PipelineStageFlagBits::eTopOfPipe,
			vk::DependencyFlags(),
			0, nullptr,
			0, nullptr,
			1, &barrier);

		endSingleTimeCommands(commandBuffer);
	}

	void copyBufferToImage(vk::Buffer buffer, vk::Image image, uint32_t width, uint32_t height)
	{
		vk::CommandBuffer commandBuffer = beginSingleTimeCommands();

		vk::ImageSubresourceLayers subresourceLayers = vk::ImageSubresourceLayers()
			.setAspectMask(vk::ImageAspectFlagBits::eColor)
			.setMipLevel(0)
			.setBaseArrayLayer(0)
			.setLayerCount(1);

		vk::BufferImageCopy region = vk::BufferImageCopy()
			.setBufferOffset(0)
			.setBufferRowLength(0)
			.setBufferImageHeight(0)
			.setImageSubresource(subresourceLayers)
			.setImageOffset({ 0, 0,0 })
			.setImageExtent({ width, height, 1 });

		commandBuffer.copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, 1, &region);

		endSingleTimeCommands(commandBuffer);
	}

	void createVertexBuffer()
	{
		vk::DeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

		vk::Buffer stagingBuffer;
		vk::DeviceMemory stagingBufferMemory;
		createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

		void* data;
		device.mapMemory(stagingBufferMemory, 0, bufferSize, vk::MemoryMapFlags(), &data);
		memcpy(data, vertices.data(), (size_t)bufferSize);
		device.unmapMemory(stagingBufferMemory);
		
		createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal, vertexBuffer, vertexMemory);

		copyBuffer(stagingBuffer, vertexBuffer, bufferSize);

		device.destroyBuffer(stagingBuffer);
		device.freeMemory(stagingBufferMemory);
	}

	void createIndexBuffer()
	{
		vk::DeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

		vk::Buffer stagingBuffer;
		vk::DeviceMemory stagingBufferMemory;
		createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

		void* data;
		device.mapMemory(stagingBufferMemory, 0, bufferSize, vk::MemoryMapFlags(), &data);
		memcpy(data, indices.data(), (size_t)bufferSize);
		device.unmapMemory(stagingBufferMemory);

		createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal, indexBuffer, indexMemory);

		copyBuffer(stagingBuffer, indexBuffer, bufferSize);

		device.destroyBuffer(stagingBuffer);
		device.freeMemory(stagingBufferMemory);
	}

	void createUniformBuffer()
	{
		vk::DeviceSize bufferSize = sizeof(uniformBufferObject);
		createBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer, (vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent), uniformBuffer, uniformMemory);
	}

	void createDescriptorPool()
	{
		std::vector<vk::DescriptorPoolSize> poolsizes(2);
		poolsizes[0].setType(vk::DescriptorType::eUniformBuffer)
			.setDescriptorCount(1);
		poolsizes[1].setType(vk::DescriptorType::eCombinedImageSampler)
			.setDescriptorCount(1);

		vk::DescriptorPoolCreateInfo poolInfo = vk::DescriptorPoolCreateInfo()
			.setPoolSizeCount(static_cast<uint32_t>(poolsizes.size()))
			.setPPoolSizes(poolsizes.data())
			.setMaxSets(1);

		descriptorPool = device.createDescriptorPool(poolInfo);
	}

	void createDescriptorSet()
	{
		vk::DescriptorSetLayout layouts[] = { descriptorSetLayout };
		vk::DescriptorSetAllocateInfo allocateInfo = vk::DescriptorSetAllocateInfo()
			.setDescriptorPool(descriptorPool)
			.setDescriptorSetCount(1)
			.setPSetLayouts(layouts);

		device.allocateDescriptorSets(&allocateInfo, &descriptorSet);

		vk::DescriptorBufferInfo bufferInfo = vk::DescriptorBufferInfo()
			.setBuffer(uniformBuffer)
			.setOffset(0)
			.setRange(sizeof(uniformBufferObject));

		vk::DescriptorImageInfo imageInfo = vk::DescriptorImageInfo()
			.setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
			.setImageView(textureImageView)
			.setSampler(textureSampler);

		std::vector<vk::WriteDescriptorSet> descriptorWrites(2);
		descriptorWrites[0].setDstSet(descriptorSet)
			.setDstBinding(0)
			.setDstArrayElement(0)
			.setDescriptorType(vk::DescriptorType::eUniformBuffer)
			.setDescriptorCount(1)
			.setPBufferInfo(&bufferInfo);

		descriptorWrites[1].setDstSet(descriptorSet)
			.setDstBinding(1)
			.setDstArrayElement(0)
			.setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
			.setDescriptorCount(1)
			.setPImageInfo(&imageInfo);

		device.updateDescriptorSets(descriptorWrites, 0);
	}

	void createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, vk::Buffer& buffer, vk::DeviceMemory& memory)
	{
		vk::BufferCreateInfo bufferInfo = vk::BufferCreateInfo()
			.setSize(size)
			.setUsage(usage)
			.setSharingMode(vk::SharingMode::eExclusive);

		buffer = device.createBuffer(bufferInfo);

		vk::MemoryRequirements memRequirements = device.getBufferMemoryRequirements(buffer);

		vk::MemoryAllocateInfo allocateInfo = vk::MemoryAllocateInfo()
			.setAllocationSize(memRequirements.size)
			.setMemoryTypeIndex(findMemoryType(memRequirements.memoryTypeBits, properties));

		memory = device.allocateMemory(allocateInfo);

		device.bindBufferMemory(buffer, memory, 0);
	}

	vk::CommandBuffer beginSingleTimeCommands()
	{
		vk::CommandBufferAllocateInfo allocateInfo = vk::CommandBufferAllocateInfo()
			.setLevel(vk::CommandBufferLevel::ePrimary)
			.setCommandPool(commandPool)
			.setCommandBufferCount(1);

		vk::CommandBuffer commandBuffer; 
		device.allocateCommandBuffers(&allocateInfo, &commandBuffer);

		vk::CommandBufferBeginInfo beginInfo = vk::CommandBufferBeginInfo()
			.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

		commandBuffer.begin(beginInfo);

		return commandBuffer;
	}

	void endSingleTimeCommands(vk::CommandBuffer commandBuffer)
	{
		commandBuffer.end();

		vk::SubmitInfo submitInfo = vk::SubmitInfo()
			.setCommandBufferCount(1)
			.setPCommandBuffers(&commandBuffer);

		graphicsQueue.submit(submitInfo, VK_NULL_HANDLE);
		graphicsQueue.waitIdle();

		device.freeCommandBuffers(commandPool, commandBuffer);
	}

	void copyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size)
	{
		vk::CommandBuffer commandBuffer = beginSingleTimeCommands();

		vk::BufferCopy copyRegion = vk::BufferCopy()
			.setSize(size);
		commandBuffer.copyBuffer(srcBuffer, dstBuffer, copyRegion);

		endSingleTimeCommands(commandBuffer);
	}

	uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties)
	{
		vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();

		for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
		{
			if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
			{
				return i;
			}
		}

		throw std::runtime_error("Failed to find memory type!");
	}

	void createCommandBuffers()
	{
		commandBuffers.resize(swapchainFramebuffers.size());

		vk::CommandBufferAllocateInfo allocateInfo = vk::CommandBufferAllocateInfo()
			.setCommandPool(commandPool)
			.setLevel(vk::CommandBufferLevel::ePrimary)
			.setCommandBufferCount((uint32_t)commandBuffers.size());

		commandBuffers = device.allocateCommandBuffers(allocateInfo);

		for (size_t i = 0; i < commandBuffers.size(); i++)
		{
			vk::CommandBufferBeginInfo beginInfo = vk::CommandBufferBeginInfo()
				.setFlags(vk::CommandBufferUsageFlagBits::eSimultaneousUse);

			commandBuffers[i].begin(beginInfo);

			vk::RenderPassBeginInfo renderPassInfo = vk::RenderPassBeginInfo()
				.setRenderPass(renderPass)
				.setFramebuffer(swapchainFramebuffers[i])
				.setRenderArea(vk::Rect2D({ 0, 0 }, swapchainExtent));

			std::vector<vk::ClearValue> clearValues(2);
			vk::ClearColorValue ccv = vk::ClearColorValue()
			.setFloat32({ 0.0f, 0.0f, 0.0f, 1.0f });
			clearValues[0].setColor(ccv);
			clearValues[1].setDepthStencil({ 1.0f, 0 });

			renderPassInfo.setClearValueCount(static_cast<uint32_t>(clearValues.size()))
				.setPClearValues(clearValues.data());

			commandBuffers[i].beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
			commandBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline);
			vk::Buffer vertexBuffers[] = { vertexBuffer };
			vk::DeviceSize offsets[] = { 0 };
			commandBuffers[i].bindVertexBuffers(0, 1, vertexBuffers, offsets);
			commandBuffers[i].bindIndexBuffer(indexBuffer, 0, vk::IndexType::eUint16);
			commandBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
			commandBuffers[i].drawIndexed(static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
			commandBuffers[i].endRenderPass();

			commandBuffers[i].end();
		}
	}

	void createSemaphores()
	{
		vk::SemaphoreCreateInfo semaphoreInfo;

		imageAvailableSemaphore = device.createSemaphore(semaphoreInfo);
		renderFinishedSemaphore = device.createSemaphore(semaphoreInfo);
	}

	void updateUniformBuffer()
	{
		static auto startTime = std::chrono::high_resolution_clock::now();

		auto currentTime = std::chrono::high_resolution_clock::now();
		float time = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - startTime).count() / 1000.0f;

		uniformBufferObject ubo;
		ubo.model = glm::rotate(glm::mat4(), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
		ubo.view = glm::lookAt(glm::vec3(-2.0f, -2.0f, -2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f));
		ubo.proj = glm::perspective(glm::radians(45.0f), swapchainExtent.width / (float)swapchainExtent.height, 0.1f, 10.0f);
		ubo.proj[1][1] *= -1;

		void* data;
		device.mapMemory(uniformMemory, 0, sizeof(ubo), vk::MemoryMapFlags(), &data);
		memcpy(data, &ubo, sizeof(ubo));
		device.unmapMemory(uniformMemory);

	}

	void drawFrame()
	{
		uint32_t imageIndex;
		vk::Result res = device.acquireNextImageKHR(swapchain, UINT64_MAX, imageAvailableSemaphore, VK_NULL_HANDLE, &imageIndex);

		if (res == vk::Result::eErrorOutOfDateKHR)
		{
			recreateSwapchain();
			return;
		}
		else if (res != vk::Result::eSuccess && res != vk::Result::eSuboptimalKHR)
		{
			throw std::runtime_error("Failed to acquire swapchain image!");
		}

		vk::SubmitInfo submitInfo;

		vk::Semaphore waitSemaphores[] = { imageAvailableSemaphore };
		vk::PipelineStageFlags waitStages[] = { vk::PipelineStageFlagBits::eColorAttachmentOutput };
		submitInfo.setWaitSemaphoreCount(1)
			.setPWaitSemaphores(waitSemaphores)
			.setPWaitDstStageMask(waitStages)
			.setCommandBufferCount(1)
			.setPCommandBuffers(&commandBuffers[imageIndex]);

		vk::Semaphore signalSemaphores[] = { renderFinishedSemaphore };
		submitInfo.setSignalSemaphoreCount(1)
			.setPSignalSemaphores(signalSemaphores);

		graphicsQueue.submit(submitInfo, VK_NULL_HANDLE);

		vk::PresentInfoKHR presentInfo = vk::PresentInfoKHR()
			.setWaitSemaphoreCount(1)
			.setPWaitSemaphores(signalSemaphores);

		vk::SwapchainKHR swapchains[] = { swapchain };
		presentInfo.setSwapchainCount(1)
			.setPSwapchains(swapchains)
			.setPImageIndices(&imageIndex);

		res = presentQueue.presentKHR(presentInfo);

		if (res == vk::Result::eErrorOutOfDateKHR || res == vk::Result::eSuboptimalKHR)
		{
			recreateSwapchain();
		}
		else if (res != vk::Result::eSuccess)
		{
			throw std::runtime_error("Failed to present swapchain image!");
		}

		presentQueue.waitIdle();
	}

	vk::ShaderModule createShaderModule(const std::vector<char>& code)
	{
		vk::ShaderModuleCreateInfo createInfo = vk::ShaderModuleCreateInfo()
			.setCodeSize(code.size())
			.setPCode(reinterpret_cast<const uint32_t*>(code.data()));

		vk::ShaderModule shaderModule = device.createShaderModule(createInfo);

		return shaderModule;
	}

	vk::SurfaceFormatKHR chooseSwapchainSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats)
	{
		if (availableFormats.size() == 1 && availableFormats[0].format == vk::Format::eUndefined)
		{
			return{ vk::Format::eB8G8R8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear };
		}

		for (const auto& availableFormat : availableFormats)
		{
			if (availableFormat.format == vk::Format::eB8G8R8A8Unorm && availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
			{
				return availableFormat;
			}
		}

		return availableFormats[0];
	}

	vk::PresentModeKHR chooseSwapchainPresentMode(const std::vector<vk::PresentModeKHR> availablepresentModes)
	{
		vk::PresentModeKHR bestMode = vk::PresentModeKHR::eFifo;

		for (const auto& availablepresentMode : availablepresentModes)
		{
			if (availablepresentMode == vk::PresentModeKHR::eMailbox)
			{
				return availablepresentMode;
			}
			else if (availablepresentMode == vk::PresentModeKHR::eImmediate)
			{
				bestMode = availablepresentMode;
			}
		}

		return bestMode;
	}

	vk::Extent2D chooseSwapchainExtent(const vk::SurfaceCapabilitiesKHR& capabilities)
	{
		if (capabilities.currentExtent.width != UINT32_MAX)
		{
			return capabilities.currentExtent;
		}
		else
		{
			int width, height;
			SDL_GetWindowSize(window, &width, &height);

			vk::Extent2D actualExtent =
			{
				static_cast<uint32_t>(width),
				static_cast<uint32_t>(height)
			};

			actualExtent.width = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actualExtent.width));
			actualExtent.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, actualExtent.height));

			return actualExtent;
		}
	}

	swapchainDetails querySwapchainSupport(vk::PhysicalDevice device)
	{
		swapchainDetails details;

		details.capabilities = device.getSurfaceCapabilitiesKHR(surface);

		details.formats = device.getSurfaceFormatsKHR(surface);

		details.presentModes = device.getSurfacePresentModesKHR(surface);

		return details;
	}

	bool isDeviceSuitable(vk::PhysicalDevice device)
	{
		queueFamilyIndices indices = findQueueFamilies(device);

		bool extensionsSupported = checkDeviceExtensionSupport(device);

		bool swapchainAdequate = false;
		if (extensionsSupported)
		{
			swapchainDetails swapchainSupport = querySwapchainSupport(device);
			swapchainAdequate = !swapchainSupport.formats.empty() && !swapchainSupport.presentModes.empty();
		}

		vk::PhysicalDeviceFeatures supportedFeatures = device.getFeatures();

		return indices.isComplete() && extensionsSupported && supportedFeatures.samplerAnisotropy;
	}

	bool checkDeviceExtensionSupport(vk::PhysicalDevice device)
	{
		std::vector<vk::ExtensionProperties> availableExtensions = device.enumerateDeviceExtensionProperties();

		std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());
		
		for (const auto& extension : availableExtensions)
		{
			requiredExtensions.erase(extension.extensionName);
		}

		return requiredExtensions.empty();
	}

	queueFamilyIndices findQueueFamilies(vk::PhysicalDevice device)
	{
		queueFamilyIndices indices;

		std::vector<vk::QueueFamilyProperties> queueFamilies = device.getQueueFamilyProperties();

		int i = 0;
		for (const auto& queueFamily : queueFamilies)
		{
			if (queueFamily.queueCount > 0 && queueFamily.queueFlags & vk::QueueFlagBits::eGraphics)
			{
				indices.graphicsIndex = i;
			}

			vk::Bool32 presentSupport = false;
			presentSupport = device.getSurfaceSupportKHR(i, surface);

			if (queueFamily.queueCount > 0 && presentSupport)
			{
				indices.presentIndex = i;
			}

			if(indices.isComplete())
			{
				break;
			}

			i++;
		}

		return indices;
	}

	std::vector<const char*> getRequiredExtensions()
	{
		std::vector<const char*> extensions;

		extensions = getAvailableWSIExtensions();

		if (enableValidationLayers)
		{
			extensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
		}

		return extensions;
	}

	bool checkValidationLayerSupport()
	{
		uint32_t layerCount;
		vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

		std::vector<VkLayerProperties> availableLayers(layerCount);
		vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

		for (const char* layerName : validationLayers) {
			bool layerFound = false;

			for (const auto& layerProperties : availableLayers) {
				if (strcmp(layerName, layerProperties.layerName) == 0) {
					layerFound = true;
					break;
				}
			}

			if (!layerFound) {
				return false;
			}
		}

		return true;
	}

	static std::vector<char> readFile(const std::string& filename)
	{
		TCHAR nPath[MAX_PATH + 1];
		GetCurrentDirectory(MAX_PATH, nPath);
		std::string correctedName = std::string(nPath) + filename;
		//std::cout << correctedName << '\n';
		std::ifstream file(correctedName, std::ios::ate | std::ios::binary);

		if (!file.is_open()) {
			throw std::runtime_error("Failed to open file: " + filename +"!");
		}

		size_t fileSize = (size_t)file.tellg();
		std::vector<char> buffer(fileSize);

		file.seekg(0);
		file.read(buffer.data(), fileSize);

		file.close();

		return buffer;
	}
};

int main()
{
	VulkanTestV2 app;

	try {
		app.run();
	}
	catch (const std::runtime_error& e)
	{
		std::cerr << e.what() << '\n';
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}