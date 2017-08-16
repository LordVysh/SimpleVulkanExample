/*
 * Vulkan Windowed Program
 *
 * Copyright (C) 2016 Valve Corporation
 * Copyright (C) 2016 LunarG, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
Vulkan C++ Windowed Project Template
Create and destroy a Vulkan surface on an SDL window.
*/


// Enable the WSI extensions
#if defined(__ANDROID__)
#define VK_USE_PLATFORM_ANDROID_KHR
#elif defined(__linux__)
#define VK_USE_PLATFORM_XLIB_KHR
#elif defined(_WIN32)
#define VK_USE_PLATFORM_WIN32_KHR
#endif

// Tell SDL not to mess with main()
#define SDL_MAIN_HANDLED

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <SDL2/SDL.h>
#include <SDL2/SDL_syswm.h>
#include <vulkan/vulkan.hpp>

#include <iostream>
#include <vector>
#include <fstream>

#include "cube_data.h"

vk::SurfaceKHR createVulkanSurface(const vk::Instance& instance, SDL_Window* window);
std::vector<const char*> getAvailableWSIExtensions();
void vkCreateDebugReportCallbackEXT(vk::Instance& instance, vk::DebugReportCallbackCreateInfoEXT* pCreateInfo, vk::AllocationCallbacks* pAllocatorCallbacks, vk::DebugReportCallbackEXT* pCallback);
void vkDestroyDebugReportCallbackEXT(vk::Instance& instance, vk::DebugReportCallbackEXT* callback, vk::AllocationCallbacks* pAllocatorCallbacks);
vk::Extent3D createExtent3DFromSurfaceCapabilities(vk::SurfaceCapabilitiesKHR& surfaceCapabilities);
static std::vector<char> readFile(const std::string& filename);
VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugReportFlagsEXT flags, VkDebugReportObjectTypeEXT objType, uint64_t obj, size_t location, int32_t code, const char* layerPrefix, const char* msg, void* userData);

int main()
{
	// Use validation layers if this is a debug build, and use WSI extensions regardless
	std::vector<const char*> extensions = getAvailableWSIExtensions();
	std::vector<const char*> layers;
#if defined(_DEBUG)
	layers.push_back("VK_LAYER_LUNARG_standard_validation");
	extensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
#endif

	// vk::ApplicationInfo allows the programmer to specifiy some basic information about the
	// program, which can be useful for layers and tools to provide more debug information.
	vk::ApplicationInfo appInfo = vk::ApplicationInfo()
		.setPApplicationName("Vulkan C++ Windowed Program Template")
		.setApplicationVersion(1)
		.setPEngineName("LunarG SDK")
		.setEngineVersion(1)
		.setApiVersion(VK_API_VERSION_1_0);

	// vk::InstanceCreateInfo is where the programmer specifies the layers and/or extensions that
	// are needed.
	vk::InstanceCreateInfo instInfo = vk::InstanceCreateInfo()
		.setFlags(vk::InstanceCreateFlags())
		.setPApplicationInfo(&appInfo)
		.setEnabledExtensionCount(static_cast<uint32_t>(extensions.size()))
		.setPpEnabledExtensionNames(extensions.data())
		.setEnabledLayerCount(static_cast<uint32_t>(layers.size()))
		.setPpEnabledLayerNames(layers.data());

	// Create the Vulkan instance.
	vk::Instance instance;
	try {
		instance = vk::createInstance(instInfo);
	}
	catch (const std::exception& e) {
		std::cout << "Could not create a Vulkan instance: " << e.what() << std::endl;
		return 1;
	}

	// Create an SDL window that supports Vulkan and OpenGL rendering.
	if (SDL_Init(SDL_INIT_VIDEO) != 0) {
		std::cout << "Could not initialize SDL." << std::endl;
		return 1;
	}
	SDL_Window* window = SDL_CreateWindow("Vulkan Window", SDL_WINDOWPOS_CENTERED,
		SDL_WINDOWPOS_CENTERED, 1280, 720, SDL_WINDOW_OPENGL);
	if (window == NULL) {
		std::cout << "Could not create SDL window." << std::endl;
		return 1;
	}

	// Create a Vulkan surface for rendering
	vk::SurfaceKHR surface;
	try {
		surface = createVulkanSurface(instance, window);
	}
	catch (const std::exception& e) {
		std::cout << "Failed to create Vulkan surface: " << e.what() << std::endl;
		instance.destroy();
		return 1;
	}

	// This is where most initializtion for a program should be performed

	//Create debug output for validation layers
	vk::DebugReportCallbackCreateInfoEXT debugReportCallbackCreateInfo = vk::DebugReportCallbackCreateInfoEXT()
		.setFlags(vk::DebugReportFlagBitsEXT::eError | vk::DebugReportFlagBitsEXT::eWarning)
		.setPfnCallback(debugCallback);
	vk::DebugReportCallbackEXT callback;
	vkCreateDebugReportCallbackEXT(instance, &debugReportCallbackCreateInfo, nullptr, &callback);

	//Enumerate the physical devices on the system
	std::vector<vk::PhysicalDevice> physicalDevices = instance.enumeratePhysicalDevices();

	//Create a struct to hold device queue creation info
	vk::DeviceQueueCreateInfo deviceQueueCreateInfo[2];

	//Create a vector to hold the properties of the device queue families on the device
	std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevices[0].getQueueFamilyProperties();

	//Iterate over the queue family properties to find a graphics capable queue family
	bool foundGraphicsQueue = false;
	bool foundPresentQueue = false;
	uint32_t presentQueue = UINT32_MAX;
	uint32_t graphicsQueue = UINT32_MAX;
	for (uint32_t i = 0; i < queueFamilyProperties.size(); ++i)
	{
		if (queueFamilyProperties[i].queueCount > 0 && queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eGraphics)
		{
			if (graphicsQueue == UINT32_MAX)
			{
				graphicsQueue = i;
				deviceQueueCreateInfo[0].setQueueFamilyIndex(i);
				foundGraphicsQueue = true;
			}
			if (physicalDevices[0].getSurfaceSupportKHR(i, surface) == VK_TRUE)
			{
				graphicsQueue = i;
				deviceQueueCreateInfo[0].setQueueFamilyIndex(i);
				presentQueue = i;
				deviceQueueCreateInfo[1].setQueueFamilyIndex(i);
				foundGraphicsQueue = true;
				foundPresentQueue = true;
				break;
			}
		}
	}
	if (presentQueue == UINT32_MAX)
	{
		for (uint32_t i = 0; i < queueFamilyProperties.size(); ++i)
		{
			if (physicalDevices[0].getSurfaceSupportKHR(i, surface) == VK_TRUE)
			{
				presentQueue = i;
				deviceQueueCreateInfo[1].setQueueFamilyIndex(i);
				foundPresentQueue = true;
				break;
			}
		}
	}
	if (foundGraphicsQueue == false)
	{
		throw std::runtime_error("Unable to find appropriate queue family for graphics operations!");
	}
	if (foundPresentQueue == false)
	{
		throw std::runtime_error("Unable to find appropriate queue family for present operations!");
	}

	//Finish populating the device queue create information
	float queue_priorities[1] = { 0.0 };
	deviceQueueCreateInfo[0].setQueueCount(1)
		.setPQueuePriorities(queue_priorities);
	deviceQueueCreateInfo[1].setQueueCount(1)
		.setPQueuePriorities(queue_priorities);

	//Poll for device features and enable depth clamping
	vk::PhysicalDeviceFeatures availableFeatures = physicalDevices[0].getFeatures();
	vk::PhysicalDeviceFeatures enabledFeatures;
	if (availableFeatures.depthClamp == VK_TRUE)
		enabledFeatures.setDepthClamp(VK_TRUE);

	//Populate information needed to create a logical device
	std::vector<const char*> deviceExtensions;
	deviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
	vk::DeviceCreateInfo deviceCreateinfo = vk::DeviceCreateInfo()
		.setQueueCreateInfoCount(1)
		.setPQueueCreateInfos(deviceQueueCreateInfo)
		.setEnabledExtensionCount(static_cast<uint32_t>(deviceExtensions.size()))
		.setPpEnabledExtensionNames(deviceExtensions.data())
		.setPEnabledFeatures(&enabledFeatures)
		.setEnabledLayerCount(0)
		.setPpEnabledLayerNames(NULL);
	if (presentQueue != graphicsQueue)
	{
		deviceCreateinfo.setQueueCreateInfoCount(2);
	}

	//Create the logical device
	vk::Device device = physicalDevices[0].createDevice(deviceCreateinfo);
	if (device == NULL)
	{
		throw std::runtime_error("Failed to create device!");
	}

	//Get the format used by the surface
	vk::Format swapchainFormat;
	std::vector<vk::SurfaceFormatKHR> surfaceFormats = physicalDevices[0].getSurfaceFormatsKHR(surface);
	//If the surface has no preferred format set it to something
	if (surfaceFormats.size() == 1 && surfaceFormats[0].format == vk::Format::eUndefined)
	{
		swapchainFormat = vk::Format::eB8G8R8A8Unorm;
	}
	//Otherwise allow it to use it's preferred format
	else
	{
		swapchainFormat = surfaceFormats[0].format;
	}

	//Start populating the information needed to create a swapchain
	vk::SwapchainCreateInfoKHR swapchainCreateInfo = vk::SwapchainCreateInfoKHR()
		.setSurface(surface)
		.setImageFormat(swapchainFormat);
	//Get surface capabilities
	vk::SurfaceCapabilitiesKHR surfaceCapabilities = physicalDevices[0].getSurfaceCapabilitiesKHR(surface);
	uint32_t desiredNumberOfSwapChainImages = surfaceCapabilities.minImageCount;
	swapchainCreateInfo.setMinImageCount(desiredNumberOfSwapChainImages);

	//Get surface present modes
	std::vector<vk::PresentModeKHR> presentModes = physicalDevices[0].getSurfacePresentModesKHR(surface);

	//Create swapchain extent based on surface
	vk::Extent2D swapchainExtent;
	if (surfaceCapabilities.currentExtent.width == 0xFFFFFFFF)
	{
		// If the surface size is undefined, the size is set to
		// the size of the images requested.
		if (swapchainExtent.width < surfaceCapabilities.minImageExtent.width) 
		{
			swapchainExtent.width = surfaceCapabilities.minImageExtent.width;
		}
		else if (swapchainExtent.width > surfaceCapabilities.maxImageExtent.width) 
		{
			swapchainExtent.width = surfaceCapabilities.maxImageExtent.width;
		}

		if (swapchainExtent.height < surfaceCapabilities.minImageExtent.height) 
		{
			swapchainExtent.height = surfaceCapabilities.minImageExtent.height;
		}
		else if (swapchainExtent.height > surfaceCapabilities.maxImageExtent.height) 
		{
			swapchainExtent.height = surfaceCapabilities.maxImageExtent.height;
		}
	}

	else 
	{
		// If the surface size is defined, the swap chain size must match
		swapchainExtent = surfaceCapabilities.currentExtent;
	}

	//Set swapchain image extent and define the present mode
	swapchainCreateInfo.setImageExtent(swapchainExtent);
	swapchainCreateInfo.setPresentMode(vk::PresentModeKHR::eFifo);

	//Set up the preTransform based on the surface and set it
	vk::SurfaceTransformFlagBitsKHR preTransform;
	if (surfaceCapabilities.supportedTransforms & vk::SurfaceTransformFlagBitsKHR::eIdentity) 
	{
		preTransform = vk::SurfaceTransformFlagBitsKHR::eIdentity;
	}
	else 
	{
		preTransform = surfaceCapabilities.currentTransform;
	}
	swapchainCreateInfo.setPreTransform(preTransform);

	//Set even more stuff
	swapchainCreateInfo.setImageSharingMode(vk::SharingMode::eExclusive);
	swapchainCreateInfo.setQueueFamilyIndexCount(0);
	swapchainCreateInfo.setPQueueFamilyIndices(NULL);
	swapchainCreateInfo.setImageArrayLayers(1);
	swapchainCreateInfo.setOldSwapchain(VK_NULL_HANDLE);
#ifndef __ANDROID__
	swapchainCreateInfo.clipped = true;
#else
	swapchainCreateInfo.clipped = false;
#endif
	swapchainCreateInfo.setImageColorSpace(vk::ColorSpaceKHR::eSrgbNonlinear);
	vk::ImageUsageFlags usageFlags = (vk::ImageUsageFlagBits::eColorAttachment |
		vk::ImageUsageFlagBits::eTransferSrc);
	swapchainCreateInfo.setImageUsage(usageFlags);

	uint32_t queueFamilyIndices[2] =
	{
		(uint32_t)graphicsQueue,
		(uint32_t)presentQueue
	};
	if (graphicsQueue != presentQueue)
	{
		// If the graphics and present queues are from different queue families,
		// we either have to explicitly transfer ownership of images between the
		// queues, or we have to create the swapchain with imageSharingMode
		// as vk::SharingMode::eConcurrent
		swapchainCreateInfo.setImageSharingMode(vk::SharingMode::eConcurrent);
		swapchainCreateInfo.setQueueFamilyIndexCount(2);
		swapchainCreateInfo.setPQueueFamilyIndices(queueFamilyIndices);
	}

	//Determine swapchain composite alpha
	vk::CompositeAlphaFlagBitsKHR compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
	vk::CompositeAlphaFlagBitsKHR compositeAlphaFlags[4] = {
		vk::CompositeAlphaFlagBitsKHR::eOpaque,
		vk::CompositeAlphaFlagBitsKHR::ePreMultiplied,
		vk::CompositeAlphaFlagBitsKHR::ePostMultiplied,
		vk::CompositeAlphaFlagBitsKHR::eInherit,
	};
	for (uint32_t i = 0; i < sizeof(compositeAlphaFlags); i++) {
		if (surfaceCapabilities.supportedCompositeAlpha & compositeAlphaFlags[i]) {
			compositeAlpha = compositeAlphaFlags[i];
			break;
		}
	}
	swapchainCreateInfo.setCompositeAlpha(compositeAlpha);

	//FINALLY create the swapchain
	vk::SwapchainKHR swapchain = device.createSwapchainKHR(swapchainCreateInfo);

	//Get swapchain images
	std::vector<vk::Image> swapchainImages = device.getSwapchainImagesKHR(swapchain);
	std::vector<vk::ImageView> swapchainImageViews;
	vk::ComponentMapping componentMapping = vk::ComponentMapping();
	vk::ImageSubresourceRange subresourceRange = vk::ImageSubresourceRange()
		.setAspectMask(vk::ImageAspectFlagBits::eColor)
		.setLevelCount(1)
		.setLayerCount(1);

	for (uint32_t i = 0; i < swapchainImages.size(); i++)
	{
		vk::ImageViewCreateInfo imageViewCreateInfo = vk::ImageViewCreateInfo();
		imageViewCreateInfo.setFormat(swapchainFormat);
		imageViewCreateInfo.setComponents(componentMapping);
		imageViewCreateInfo.setSubresourceRange(subresourceRange);
		imageViewCreateInfo.setViewType(vk::ImageViewType::e2D);

		imageViewCreateInfo.setImage(swapchainImages[i]);
		swapchainImageViews.push_back(device.createImageView(imageViewCreateInfo));
	}


	//Create a depth buffer image
	vk::Extent3D depthBufferExtent = createExtent3DFromSurfaceCapabilities(surfaceCapabilities);
	depthBufferExtent.setDepth(1);
	vk::ImageCreateInfo depthBufferCreateInfo = vk::ImageCreateInfo()
		.setImageType(vk::ImageType::e2D)
		.setFormat(vk::Format::eD16Unorm)
		.setExtent(depthBufferExtent)
		.setMipLevels(1)
		.setArrayLayers(1)
		.setSamples(vk::SampleCountFlagBits::e1)
		.setInitialLayout(vk::ImageLayout::eUndefined)
		.setUsage(vk::ImageUsageFlagBits::eDepthStencilAttachment)
		.setQueueFamilyIndexCount(0)
		.setPQueueFamilyIndices(NULL)
		.setSharingMode(vk::SharingMode::eExclusive);

	vk::Image depthBufferImage = device.createImage(depthBufferCreateInfo);

	//Allocate memory for the depth buffer
	vk::MemoryRequirements depthBufferMemoryRequirements = device.getImageMemoryRequirements(depthBufferImage);
	vk::PhysicalDeviceMemoryProperties physicalDeviceMemoryProperties = physicalDevices[0].getMemoryProperties();
	uint32_t typeIndex;
	for (uint32_t i = 0; i < physicalDeviceMemoryProperties.memoryTypeCount; i++) {
		if ((depthBufferMemoryRequirements.memoryTypeBits & 1) == 1) {
			// Type is available, does it match user properties?
			if ((physicalDeviceMemoryProperties.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal) == vk::MemoryPropertyFlagBits::eDeviceLocal) {
				typeIndex = i;
			}
		}
		depthBufferMemoryRequirements.memoryTypeBits >>= 1;
	}
	vk::MemoryAllocateInfo depthBufferAllocateInfo = vk::MemoryAllocateInfo()
		.setAllocationSize(depthBufferMemoryRequirements.size)
		.setMemoryTypeIndex(typeIndex);

	vk::DeviceMemory depthBufferMemory = device.allocateMemory(depthBufferAllocateInfo);
	device.bindImageMemory(depthBufferImage, depthBufferMemory, 0);

	//Create image view for depth buffer
	vk::ComponentMapping depthBufferComponentMapping = vk::ComponentMapping();
	vk::ImageSubresourceRange depthBufferSubresourceRange = vk::ImageSubresourceRange()
		.setAspectMask(vk::ImageAspectFlagBits::eDepth)
		.setBaseMipLevel(0)
		.setLevelCount(1)
		.setBaseArrayLayer(0)
		.setLayerCount(1);
	vk::ImageViewCreateInfo depthBufferImageViewCreateInto = vk::ImageViewCreateInfo()
		.setImage(depthBufferImage)
		.setFormat(vk::Format::eD16Unorm)
		.setComponents(depthBufferComponentMapping)
		.setSubresourceRange(depthBufferSubresourceRange)
		.setViewType(vk::ImageViewType::e2D);

	vk::ImageView depthBufferImageView = device.createImageView(depthBufferImageViewCreateInto);

	//Create MVP(Model-View-Projection matrix to use with Uniform Buffer
	glm::mat4 Projection = glm::perspective(glm::radians(45.0f), 1.0f, 0.1f, 100.f);
	glm::mat4 View = glm::lookAt(
		glm::vec3(-5, 3, -10), //World position
		glm::vec3(0, 0, 0), //Look at origin
		glm::vec3(0, -1, 0) //Head is up
	);
	glm::mat4 Model = glm::mat4(1.0f);
	glm::mat4 Clip = glm::mat4(
		1.0f, 0.0f, 0.0f, 0.0f,		//Vulkan clip space has 
		0.0f, -1.0f, 0.0f, 0.0f,	//inverted Y (-1.0f)
		0.0f, 0.0f, 0.5f, 0.0f,		//and half Z (0.5f)
		0.0f, 0.0f, 0.5f, 1.0f);

	glm::mat4 MVP = Clip * Projection * View * Model;

	//Create the Uniform Buffer object
	vk::BufferCreateInfo uniformBufferCreateInfo = vk::BufferCreateInfo()
		.setUsage(vk::BufferUsageFlagBits::eUniformBuffer)
		.setSize(sizeof(MVP))
		.setQueueFamilyIndexCount(0)
		.setPQueueFamilyIndices(NULL)
		.setSharingMode(vk::SharingMode::eExclusive);

	vk::Buffer uniformBuffer = device.createBuffer(uniformBufferCreateInfo);

	//Allocate memory for the buffer
	vk::MemoryRequirements uniformBufferMemoryRequirements = device.getBufferMemoryRequirements(uniformBuffer);
	vk::MemoryPropertyFlags uniformBufferRequirementsMask = (vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
	vk::MemoryAllocateInfo uniformBufferMemoryAllocateInfo = vk::MemoryAllocateInfo()
		.setMemoryTypeIndex(0)
		.setAllocationSize(uniformBufferMemoryRequirements.size);
	// Search memtypes to find first index with those properties
	for (uint32_t i = 0; i < physicalDeviceMemoryProperties.memoryTypeCount; i++) {
		if ((uniformBufferMemoryRequirements.memoryTypeBits & 1) == 1) {
			// Type is available, does it match user properties?
			if ((physicalDeviceMemoryProperties.memoryTypes[i].propertyFlags & uniformBufferRequirementsMask) == uniformBufferRequirementsMask) {
				uniformBufferMemoryAllocateInfo.setMemoryTypeIndex(i);
			}
		}
		uniformBufferMemoryRequirements.memoryTypeBits >>= 1;
	}

	vk::DeviceMemory uniformBufferDeviceMemory = device.allocateMemory(uniformBufferMemoryAllocateInfo);

	//Create descriptor for buffer
	vk::DescriptorBufferInfo uniformBufferInfo = vk::DescriptorBufferInfo()
		.setBuffer(uniformBuffer)
		.setOffset(0)
		.setRange(sizeof(MVP));

	//Map uniform buffer memory and binding
	uint8_t *pData;
	vk::MemoryMapFlags uniformBufferMapFlags = vk::MemoryMapFlags();
	device.mapMemory(uniformBufferDeviceMemory, 0, uniformBufferMemoryRequirements.size, uniformBufferMapFlags, (void**)&pData);
	memcpy(pData, &MVP, sizeof(MVP));
	device.unmapMemory(uniformBufferDeviceMemory);
	device.bindBufferMemory(uniformBuffer, uniformBufferDeviceMemory, 0);

	//Create descriptor set layout
	vk::DescriptorSetLayoutBinding descriptorSetLayoutBinding[2];
	descriptorSetLayoutBinding[0].setBinding(0)
		.setDescriptorType(vk::DescriptorType::eUniformBuffer)
		.setDescriptorCount(1)
		.setStageFlags(vk::ShaderStageFlagBits::eVertex)
		.setPImmutableSamplers(NULL);
	descriptorSetLayoutBinding[1].setBinding(1)
		.setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
		.setDescriptorCount(1)
		.setStageFlags(vk::ShaderStageFlagBits::eFragment)
		.setPImmutableSamplers(NULL);

	std::vector<vk::DescriptorSetLayout> descriptorSetLayouts;
	vk::DescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = vk::DescriptorSetLayoutCreateInfo()
		.setBindingCount(2)
		.setPBindings(descriptorSetLayoutBinding);
	descriptorSetLayouts.resize(1);
	descriptorSetLayouts[0] = device.createDescriptorSetLayout(descriptorSetLayoutCreateInfo);

	//Create pipeline layout
	vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo = vk::PipelineLayoutCreateInfo()
		.setPushConstantRangeCount(0)
		.setPPushConstantRanges(NULL)
		.setSetLayoutCount(1)
		.setPSetLayouts(descriptorSetLayouts.data());

	vk::PipelineLayout pipelineLayout = device.createPipelineLayout(pipelineLayoutCreateInfo);

	//Create a descriptor pool
	vk::DescriptorPoolSize descriptorPoolSize[2];
	descriptorPoolSize[0].setType(vk::DescriptorType::eUniformBuffer)
		.setDescriptorCount(1);
	descriptorPoolSize[1].setType(vk::DescriptorType::eCombinedImageSampler)
		.setDescriptorCount(1);

	vk::DescriptorPoolCreateInfo descriptorPoolCreateInfo = vk::DescriptorPoolCreateInfo()
		.setMaxSets(1)
		.setPoolSizeCount(2)
		.setPPoolSizes(descriptorPoolSize)
		.setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet);

	vk::DescriptorPool descriptorPool = device.createDescriptorPool(descriptorPoolCreateInfo);

	//Allocate descriptor set
	std::vector<vk::DescriptorSet> descriptorSets;
	vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo = vk::DescriptorSetAllocateInfo()
		.setDescriptorPool(descriptorPool)
		.setDescriptorSetCount(1)
		.setPSetLayouts(descriptorSetLayouts.data());
	descriptorSets.resize(1);
	descriptorSets = device.allocateDescriptorSets(descriptorSetAllocateInfo);

	//Update descriptor sets
	vk::WriteDescriptorSet writeDescriptorSet[1];
	writeDescriptorSet[0].setDstSet(descriptorSets[0]);
	writeDescriptorSet[0].setDescriptorCount(1);
	writeDescriptorSet[0].setDescriptorType(vk::DescriptorType::eUniformBuffer);
	writeDescriptorSet[0].setPBufferInfo(&uniformBufferInfo);
	writeDescriptorSet[0].setDstArrayElement(0);
	writeDescriptorSet[0].setDstBinding(0);

	device.updateDescriptorSets(1, writeDescriptorSet, 0, NULL);

	//Create a command pool
	vk::CommandPoolCreateInfo commandPoolCreateInfo = vk::CommandPoolCreateInfo()
		.setQueueFamilyIndex(deviceQueueCreateInfo[0].queueFamilyIndex);

	vk::CommandPool commandPool = device.createCommandPool(commandPoolCreateInfo);

	//Allocate memory for command buffers
	vk::CommandBufferAllocateInfo allocateInfo = vk::CommandBufferAllocateInfo()
		.setCommandPool(commandPool)
		.setLevel(vk::CommandBufferLevel::ePrimary)
		.setCommandBufferCount(1);

	std::vector<vk::CommandBuffer> commandBuffers = device.allocateCommandBuffers(allocateInfo);

	//Create render pass
	vk::AttachmentDescription attachments[2];
	attachments[0].setFormat(swapchainFormat);
	attachments[0].setSamples(vk::SampleCountFlagBits::e1);
	attachments[0].setLoadOp(vk::AttachmentLoadOp::eClear);
	attachments[0].setStoreOp(vk::AttachmentStoreOp::eStore);
	attachments[0].setStencilLoadOp(vk::AttachmentLoadOp::eDontCare);
	attachments[0].setStencilStoreOp(vk::AttachmentStoreOp::eDontCare);
	attachments[0].setInitialLayout(vk::ImageLayout::eUndefined);
	attachments[0].setFinalLayout(vk::ImageLayout::ePresentSrcKHR);

	attachments[1].setFormat(vk::Format::eD16Unorm);
	attachments[1].setSamples(vk::SampleCountFlagBits::e1);
	attachments[1].setLoadOp(vk::AttachmentLoadOp::eClear);
	attachments[1].setStoreOp(vk::AttachmentStoreOp::eDontCare);
	attachments[1].setStencilLoadOp(vk::AttachmentLoadOp::eDontCare);
	attachments[1].setStencilStoreOp(vk::AttachmentStoreOp::eDontCare);
	attachments[1].setInitialLayout(vk::ImageLayout::eUndefined);
	attachments[1].setFinalLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal);

	vk::AttachmentReference colorReference = vk::AttachmentReference()
		.setAttachment(0)
		.setLayout(vk::ImageLayout::eColorAttachmentOptimal);

	vk::AttachmentReference depthReference = vk::AttachmentReference()
		.setAttachment(1)
		.setLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal);

	vk::SubpassDescription subpass = vk::SubpassDescription()
		.setPipelineBindPoint(vk::PipelineBindPoint::eGraphics)
		.setInputAttachmentCount(0)
		.setPInputAttachments(NULL)
		.setColorAttachmentCount(1)
		.setPColorAttachments(&colorReference)
		.setPResolveAttachments(NULL)
		.setPDepthStencilAttachment(&depthReference)
		.setPreserveAttachmentCount(0)
		.setPPreserveAttachments(NULL);

	vk::RenderPassCreateInfo renderPassCreateInfo = vk::RenderPassCreateInfo()
		.setAttachmentCount(2)
		.setPAttachments(attachments)
		.setSubpassCount(1)
		.setPSubpasses(&subpass)
		.setDependencyCount(0)
		.setPDependencies(NULL);

	vk::RenderPass renderPass = device.createRenderPass(renderPassCreateInfo);

	//Load created shader
	auto vertShaderCode = readFile("shaders/vert.spv");
	auto fragShaderCode = readFile("shaders/frag.spv");
	
	vk::ShaderModuleCreateInfo vertShaderCreateInfo = vk::ShaderModuleCreateInfo()
		.setCodeSize(vertShaderCode.size())
		.setPCode(reinterpret_cast<const uint32_t*>(vertShaderCode.data()));
	vk::ShaderModule vertShaderModule = device.createShaderModule(vertShaderCreateInfo);

	vk::ShaderModuleCreateInfo fragShaderCreateInfo = vk::ShaderModuleCreateInfo()
		.setCodeSize(fragShaderCode.size())
		.setPCode(reinterpret_cast<const uint32_t*>(fragShaderCode.data()));
	vk::ShaderModule fragShaderModule = device.createShaderModule(fragShaderCreateInfo);

	//Create framebuffers
	vk::ImageView frameBufferAttachments[2];
	frameBufferAttachments[1] = depthBufferImageView;

	vk::FramebufferCreateInfo frameBufferCreateInfo = vk::FramebufferCreateInfo()
		.setRenderPass(renderPass)
		.setAttachmentCount(2)
		.setPAttachments(frameBufferAttachments)
		.setWidth(surfaceCapabilities.currentExtent.width)
		.setHeight(surfaceCapabilities.currentExtent.height)
		.setLayers(1);

	std::vector<vk::Framebuffer> frameBuffers;
	frameBuffers.resize(swapchainImages.size());
	for (size_t i = 0; i < swapchainImages.size(); i++)
	{
		frameBufferAttachments[0] = swapchainImageViews[i];
		frameBuffers[i] = device.createFramebuffer(frameBufferCreateInfo);
	}

	//Create vertex buffer
	vk::BufferCreateInfo vertexBufferCreateInfo = vk::BufferCreateInfo()
		.setUsage(vk::BufferUsageFlagBits::eVertexBuffer)
		.setSize(sizeof(g_vb_solid_face_colors_Data))
		.setQueueFamilyIndexCount(0)
		.setPQueueFamilyIndices(NULL)
		.setSharingMode(vk::SharingMode::eExclusive);

	vk::Buffer vertexBuffer = device.createBuffer(vertexBufferCreateInfo);

	//Allocate memory for vertex buffer
	vk::MemoryRequirements vertexBufferMemoryRequirements = device.getBufferMemoryRequirements(vertexBuffer);
	vk::MemoryPropertyFlags vertexBufferRequirementsMask = (vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
	vk::MemoryAllocateInfo vertexBufferMemoryAllocateInfo = vk::MemoryAllocateInfo()
		.setAllocationSize(vertexBufferMemoryRequirements.size);

	for (uint32_t i = 0; i < physicalDeviceMemoryProperties.memoryTypeCount; i++) {
		if ((vertexBufferMemoryRequirements.memoryTypeBits & 1) == 1) {
			// Type is available, does it match user properties?
			if ((physicalDeviceMemoryProperties.memoryTypes[i].propertyFlags & vertexBufferRequirementsMask) == vertexBufferRequirementsMask) {
				vertexBufferMemoryAllocateInfo.setMemoryTypeIndex(i);
			}
		}
		vertexBufferMemoryRequirements.memoryTypeBits >>= 1;
	}

	vk::DeviceMemory vertexBufferDeviceMemory = device.allocateMemory(vertexBufferMemoryAllocateInfo);

	uint8_t *pVertexData;
	vk::MemoryMapFlags vertexBufferMapFlags = vk::MemoryMapFlags();
	device.mapMemory(vertexBufferDeviceMemory, 0, vertexBufferMemoryRequirements.size, vertexBufferMapFlags, (void**)&pVertexData);
	memcpy(pVertexData, &g_vb_solid_face_colors_Data, sizeof(g_vb_solid_face_colors_Data));
	device.unmapMemory(vertexBufferDeviceMemory);
	device.bindBufferMemory(vertexBuffer, vertexBufferDeviceMemory, 0);

	//Create vertex binding
	vk::VertexInputBindingDescription vertexInputBindingDescription = vk::VertexInputBindingDescription()
		.setBinding(0)
		.setInputRate(vk::VertexInputRate::eVertex)
		.setStride(sizeof(g_vb_solid_face_colors_Data[0]));

	//Create vertex attributes
	vk::VertexInputAttributeDescription vertexInputAttributeDescriptions[2];
	vertexInputAttributeDescriptions[0].setBinding(0);
	vertexInputAttributeDescriptions[0].setLocation(0);
	vertexInputAttributeDescriptions[0].setFormat(vk::Format::eR32G32B32A32Sfloat);
	vertexInputAttributeDescriptions[0].setOffset(0);
	vertexInputAttributeDescriptions[1].setBinding(0);
	vertexInputAttributeDescriptions[1].setLocation(1);
	vertexInputAttributeDescriptions[1].setFormat(vk::Format::eR32G32B32A32Sfloat);
	vertexInputAttributeDescriptions[1].setOffset(16);

	//Create dynamic state
	vk::DynamicState dynamicStateEnables[(int)vk::DynamicState::eStencilReference - (int)vk::DynamicState::eViewport + 1];
	memset(dynamicStateEnables, 0, sizeof dynamicStateEnables);
	vk::PipelineDynamicStateCreateInfo dynamicState = vk::PipelineDynamicStateCreateInfo()
		.setPDynamicStates(dynamicStateEnables)
		.setDynamicStateCount(0);

	//Create pipeline vertex input state
	vk::PipelineVertexInputStateCreateInfo pipelineVertexInputStateCreateInfo = vk::PipelineVertexInputStateCreateInfo()
		.setVertexAttributeDescriptionCount(1)
		.setPVertexBindingDescriptions(&vertexInputBindingDescription)
		.setVertexAttributeDescriptionCount(2)
		.setPVertexAttributeDescriptions(vertexInputAttributeDescriptions);

	//Create pipeline vertex assembly state
	vk::PipelineInputAssemblyStateCreateInfo pipelineInputAssemblyStateCreateInfo = vk::PipelineInputAssemblyStateCreateInfo()
		.setPrimitiveRestartEnable(VK_FALSE)
		.setTopology(vk::PrimitiveTopology::eTriangleList);

	//Create pipeline rasterization states
	vk::PipelineRasterizationStateCreateInfo pipelineRasterizationStateCreateInfo = vk::PipelineRasterizationStateCreateInfo()
		.setPolygonMode(vk::PolygonMode::eFill)
		.setFrontFace(vk::FrontFace::eClockwise)
		.setDepthClampEnable(VK_TRUE)
		.setRasterizerDiscardEnable(VK_FALSE)
		.setDepthBiasEnable(VK_FALSE)
		.setDepthBiasConstantFactor(0)
		.setDepthBiasClamp(0)
		.setDepthBiasSlopeFactor(0)
		.setLineWidth(1.0f);

	//Create pipeline color blend state
	vk::PipelineColorBlendStateCreateInfo pipelineColorBlendStateCreateInfo;
	vk::PipelineColorBlendAttachmentState att_state[1];
	vk::ColorComponentFlags colorComponentFlags = (vk::ColorComponentFlagBits)0xf;
	att_state[0].setColorWriteMask(colorComponentFlags);
	att_state[0].setBlendEnable(VK_FALSE);
	att_state[0].setAlphaBlendOp(vk::BlendOp::eAdd);
	att_state[0].setColorBlendOp(vk::BlendOp::eAdd);
	att_state[0].setSrcColorBlendFactor(vk::BlendFactor::eZero);
	att_state[0].setDstColorBlendFactor(vk::BlendFactor::eZero);
	att_state[0].setSrcAlphaBlendFactor(vk::BlendFactor::eZero);
	att_state[0].setDstAlphaBlendFactor(vk::BlendFactor::eZero);
	pipelineColorBlendStateCreateInfo.setAttachmentCount(1)
		.setPAttachments(att_state)
		.setLogicOpEnable(VK_FALSE)
		.setLogicOp(vk::LogicOp::eNoOp)
		.setBlendConstants({ 1.0f, 1.0f, 1.0f, 1.0f });

	//Create pipeline viewport state
	vk::PipelineViewportStateCreateInfo pipelineViewportStateCreateInfo = vk::PipelineViewportStateCreateInfo()
		.setViewportCount(1)
		.setScissorCount(1);
	dynamicStateEnables[dynamicState.dynamicStateCount++] = vk::DynamicState::eViewport;
	dynamicStateEnables[dynamicState.dynamicStateCount++] = vk::DynamicState::eScissor;
	pipelineViewportStateCreateInfo.setPViewports(NULL)
		.setPScissors(NULL);

	//Create pipeline depth stencil state
	vk::StencilOpState back = vk::StencilOpState()
		.setFailOp(vk::StencilOp::eKeep)
		.setPassOp(vk::StencilOp::eKeep)
		.setCompareOp(vk::CompareOp::eAlways)
		.setCompareMask(0)
		.setReference(0)
		.setDepthFailOp(vk::StencilOp::eKeep)
		.setWriteMask(0);
	vk::PipelineDepthStencilStateCreateInfo pipelineDepthStencilStateCreateInfo = vk::PipelineDepthStencilStateCreateInfo()
		.setDepthTestEnable(VK_TRUE)
		.setDepthWriteEnable(VK_TRUE)
		.setDepthCompareOp(vk::CompareOp::eLessOrEqual)
		.setDepthBoundsTestEnable(VK_FALSE)
		.setMinDepthBounds(0)
		.setMaxDepthBounds(0)
		.setStencilTestEnable(VK_FALSE)
		.setBack(back)
		.setFront(back);

	//Create pipeline multisample state
	vk::PipelineMultisampleStateCreateInfo pipelineMultisampleStateCreateInfo = vk::PipelineMultisampleStateCreateInfo()
		.setPSampleMask(NULL)
		.setRasterizationSamples(vk::SampleCountFlagBits::e1)
		.setSampleShadingEnable(VK_FALSE)
		.setAlphaToCoverageEnable(VK_FALSE)
		.setAlphaToOneEnable(VK_FALSE)
		.setMinSampleShading(0.0);

	//Create shader stages
	vk::PipelineShaderStageCreateInfo pipelineShaderStageCreateInfo[2];
	pipelineShaderStageCreateInfo[0].setStage(vk::ShaderStageFlagBits::eVertex);
	pipelineShaderStageCreateInfo[0].setModule(vertShaderModule);
	pipelineShaderStageCreateInfo[0].setPName("main");
	pipelineShaderStageCreateInfo[1].setStage(vk::ShaderStageFlagBits::eFragment);
	pipelineShaderStageCreateInfo[1].setModule(fragShaderModule);
	pipelineShaderStageCreateInfo[1].setPName("main");

	//Create the pipeline
	vk::GraphicsPipelineCreateInfo graphicsPipelineCreateInfo = vk::GraphicsPipelineCreateInfo()
		.setLayout(pipelineLayout)
		.setBasePipelineHandle(VK_NULL_HANDLE)
		.setBasePipelineIndex(0)
		.setPVertexInputState(&pipelineVertexInputStateCreateInfo)
		.setPInputAssemblyState(&pipelineInputAssemblyStateCreateInfo)
		.setPRasterizationState(&pipelineRasterizationStateCreateInfo)
		.setPColorBlendState(&pipelineColorBlendStateCreateInfo)
		.setPTessellationState(NULL)
		.setPMultisampleState(&pipelineMultisampleStateCreateInfo)
		.setPDynamicState(&dynamicState)
		.setPViewportState(&pipelineViewportStateCreateInfo)
		.setPDepthStencilState(&pipelineDepthStencilStateCreateInfo)
		.setPStages(pipelineShaderStageCreateInfo)
		.setStageCount(2)
		.setRenderPass(renderPass)
		.setSubpass(0);

	vk::Pipeline graphicsPipeline = device.createGraphicsPipeline(VK_NULL_HANDLE, graphicsPipelineCreateInfo);

	//Get ready to begin a render pass
	vk::ClearValue clearValue[2];
	vk::ClearColorValue clearColorValue = vk::ClearColorValue()
		.setFloat32({ 0.2f, 0.2f, 0.2f,0.2f });
	vk::ClearDepthStencilValue depthStencil = vk::ClearDepthStencilValue()
		.setDepth(1.0f)
		.setStencil(0);
	clearValue[0].setColor(clearColorValue);
	clearValue[1].setDepthStencil(depthStencil);

	//Create semaphore to get image
	vk::SemaphoreCreateInfo semaphoreCreateInfo;
	vk::Semaphore imageAcquiredSemaphore = device.createSemaphore(semaphoreCreateInfo);
	vk::Semaphore drawCompleteSemaphore = device.createSemaphore(semaphoreCreateInfo);
	vk::Semaphore imageOwnershipSemaphore = device.createSemaphore(semaphoreCreateInfo);
	vk::ResultValue<uint32_t> currentBuffer = device.acquireNextImageKHR(swapchain, UINT64_MAX, imageAcquiredSemaphore, VK_NULL_HANDLE);

	//Begin render pass
	vk::Offset2D renderOffset;
	vk::Rect2D renderArea = vk::Rect2D()
		.setExtent(surfaceCapabilities.currentExtent)
		.setOffset(renderOffset);
	vk::RenderPassBeginInfo renderPassBeginInfo = vk::RenderPassBeginInfo()
		.setRenderPass(renderPass)
		.setFramebuffer(frameBuffers[currentBuffer.value])
		.setRenderArea(renderArea)
		.setClearValueCount(2)
		.setPClearValues(clearValue);


	//Tell command buffer to begin render pass
	vk::CommandBufferBeginInfo commandBufferBeginInfo = vk::CommandBufferBeginInfo()
		.setFlags(vk::CommandBufferUsageFlagBits::eSimultaneousUse);
	commandBuffers[0].begin(commandBufferBeginInfo);

	std::vector<vk::DeviceSize> offsets;
	offsets.push_back(vk::DeviceSize(0));
	commandBuffers[0].beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);
	commandBuffers[0].bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline);
	uint32_t dynamicOffsets = 0;
	commandBuffers[0].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, 1, descriptorSets.data(), 0, NULL);
	commandBuffers[0].bindVertexBuffers(0, vertexBuffer, offsets);

#ifdef __ANDROID__
	// Disable dynamic viewport on Android. Some drive has an issue with the dynamic viewport
	// feature.
#else
	vk::Viewport viewport = vk::Viewport()
		.setHeight((float)surfaceCapabilities.currentExtent.height)
		.setWidth((float)surfaceCapabilities.currentExtent.width)
		.setMinDepth(0.0f)
		.setMaxDepth(1.0f)
		.setX(0)
		.setY(0);
	commandBuffers[0].setViewport(0, viewport);
	vk::Extent2D scissorExtent = vk::Extent2D()
		.setHeight(surfaceCapabilities.currentExtent.height)
		.setWidth(surfaceCapabilities.currentExtent.width);
	vk::Offset2D scissorOffset;
	vk::Rect2D scissor = vk::Rect2D()
		.setExtent(scissorExtent)
		.setOffset(scissorOffset);
	commandBuffers[0].setScissor(0, scissor);
#endif

	commandBuffers[0].draw(12 * 3, 1, 0, 0);
	commandBuffers[0].endRenderPass();

	if (presentQueue != graphicsQueue)
	{
		auto const image_ownership_barrier =
			vk::ImageMemoryBarrier()
			.setSrcAccessMask(vk::AccessFlags())
			.setDstAccessMask(vk::AccessFlagBits::eColorAttachmentWrite)
			.setOldLayout(vk::ImageLayout::ePresentSrcKHR)
			.setNewLayout(vk::ImageLayout::ePresentSrcKHR)
			.setSrcQueueFamilyIndex(graphicsQueue)
			.setDstQueueFamilyIndex(presentQueue)
			.setImage(swapchainImages[currentBuffer.value])
			.setSubresourceRange(vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));

		commandBuffers[0].pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
			vk::PipelineStageFlagBits::eBottomOfPipe, vk::DependencyFlagBits(), 0, nullptr, 0,
			nullptr, 1, &image_ownership_barrier);
	}

	commandBuffers[0].end();

	const vk::CommandBuffer cmd_bufs[] = { commandBuffers[0] };
	vk::FenceCreateInfo fenceCreateInfo;
	vk::Fence drawFence = device.createFence(fenceCreateInfo);

	vk::PipelineStageFlags pipe_stage_flags = vk::PipelineStageFlagBits::eColorAttachmentOutput;
	std::vector<vk::SubmitInfo> submit_info;
	submit_info.push_back(vk::SubmitInfo());
	submit_info[0].setWaitSemaphoreCount(1);
	submit_info[0].setPWaitSemaphores(&imageAcquiredSemaphore);
	submit_info[0].setPWaitDstStageMask(&pipe_stage_flags);
	submit_info[0].setCommandBufferCount(1);
	submit_info[0].setPCommandBuffers(cmd_bufs);
	submit_info[0].setSignalSemaphoreCount(1);
	submit_info[0].setPSignalSemaphores(&drawCompleteSemaphore);

	if (graphicsQueue != presentQueue)
	{
		submit_info.push_back(vk::SubmitInfo());
		submit_info[1].setPWaitDstStageMask(&pipe_stage_flags)
			.setWaitSemaphoreCount(1)
			.setPWaitSemaphores(&drawCompleteSemaphore)
			.setCommandBufferCount(1)
			.setPCommandBuffers(cmd_bufs)
			.setSignalSemaphoreCount(1)
			.setPSignalSemaphores(&imageOwnershipSemaphore);
	}

	vk::Queue graphicsVKQueue = device.getQueue(graphicsQueue, 0);
	vk::Queue presentVKQueue;
	if (presentQueue == graphicsQueue)
	{
		presentVKQueue = graphicsVKQueue;
	}
	else
	{
		presentVKQueue = device.getQueue(presentQueue, 0);
		presentVKQueue.submit(submit_info[1], vk::Fence());
	}
	graphicsVKQueue.submit(submit_info[0], drawFence);

	vk::Result res;
	do {
		res = device.waitForFences(1, &drawFence, VK_TRUE, 100000000);
	} while (res == vk::Result::eTimeout);

	device.destroyFence(drawFence);

	//Present swapchain image
	vk::PresentInfoKHR present = vk::PresentInfoKHR()
		.setSwapchainCount(1)
		.setPSwapchains(&swapchain)
		.setPImageIndices(&currentBuffer.value)
		.setPWaitSemaphores(
		(presentQueue != graphicsQueue) ? &imageOwnershipSemaphore
		: &drawCompleteSemaphore)
		.setWaitSemaphoreCount(1)
		.setPResults(NULL);
	presentVKQueue.presentKHR(present);

    // Poll for user input.
    bool stillRunning = true;
    while(stillRunning) {

        SDL_Event event;
        while(SDL_PollEvent(&event)) {

            switch(event.type) {

            case SDL_QUIT:
                stillRunning = false;
                break;

            default:
                // Do nothing.
                break;
            }
        }

        SDL_Delay(10);
    }

	//Destroy pipeline
	device.destroyPipeline(graphicsPipeline);
	//Destroy semaphore
	device.destroySemaphore(imageAcquiredSemaphore);
	device.destroySemaphore(drawCompleteSemaphore);
	device.destroySemaphore(imageOwnershipSemaphore);
	//Destroy frameBuffers
	for (auto& buffer : frameBuffers)
	{
		device.destroyFramebuffer(buffer);
	}
	//Destroy shader modules
	device.destroyShaderModule(vertShaderModule);
	device.destroyShaderModule(fragShaderModule);
	//Destroy render pass
	device.destroyRenderPass(renderPass);
	//Destroy swapchain image views
	for (uint32_t i = 0; i < swapchainImages.size(); i++)
	{
		device.destroyImageView(swapchainImageViews[i]);
	}
	//Depth buffer cleanup
	device.destroyImageView(depthBufferImageView);
	device.destroyImage(depthBufferImage);
	device.freeMemory(depthBufferMemory);
	//Uniform buffer cleanup
	device.destroyBuffer(uniformBuffer);
	device.freeMemory(uniformBufferDeviceMemory);
	//Destroy vertex buffer
	device.destroyBuffer(vertexBuffer);
	device.freeMemory(vertexBufferDeviceMemory);
	//Descriptor set layout cleanup
	device.destroyDescriptorSetLayout(descriptorSetLayouts[0]);
	//Pipeline layout cleanup
	device.destroyPipelineLayout(pipelineLayout);
	//Descriptor set cleanup
	device.freeDescriptorSets(descriptorPool, descriptorSets);
	//Descriptor pool cleanup
	device.destroyDescriptorPool(descriptorPool);
	//Swapchain cleanup
	device.destroySwapchainKHR(swapchain);
	//Cleanup command pool/buffers
	device.freeCommandBuffers(commandPool, commandBuffers);
	device.destroyCommandPool(commandPool);
	//Wait for device to idle and destroy it
	device.waitIdle();
	device.destroy();
	//Destroy surface and instance
    instance.destroySurfaceKHR(surface);
    SDL_DestroyWindow(window);
    SDL_Quit();
	vkDestroyDebugReportCallbackEXT(instance, &callback, nullptr);
    instance.destroy();

    return 0;
}

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
		<< "][Code: "
		<< code
		<< "][Message: " 
		<< msg 
		<< "]\n";
	return VK_FALSE;
}

static std::vector<char> readFile(const std::string& filename) {
	std::ifstream file(filename, std::ios::ate | std::ios::binary);

	if (!file.is_open()) {
		throw std::runtime_error("failed to open file!");
	}

	size_t fileSize = (size_t)file.tellg();
	std::vector<char> buffer(fileSize);

	file.seekg(0);
	file.read(buffer.data(), fileSize);
	file.close();
	return buffer;
}

vk::Extent3D createExtent3DFromSurfaceCapabilities(vk::SurfaceCapabilitiesKHR& surfaceCapabilities)
{	//Create swapchain extent based on surface
	vk::Extent3D extentFromSurfaceCapabilities;
	if (surfaceCapabilities.currentExtent.width == 0xFFFFFFFF)
	{
		// If the surface size is undefined, the size is set to
		// the size of the images requested.
		if (extentFromSurfaceCapabilities.width < surfaceCapabilities.minImageExtent.width)
		{
			extentFromSurfaceCapabilities.width = surfaceCapabilities.minImageExtent.width;
		}
		else if (extentFromSurfaceCapabilities.width > surfaceCapabilities.maxImageExtent.width)
		{
			extentFromSurfaceCapabilities.width = surfaceCapabilities.maxImageExtent.width;
		}

		if (extentFromSurfaceCapabilities.height < surfaceCapabilities.minImageExtent.height)
		{
			extentFromSurfaceCapabilities.height = surfaceCapabilities.minImageExtent.height;
		}
		else if (extentFromSurfaceCapabilities.height > surfaceCapabilities.maxImageExtent.height)
		{
			extentFromSurfaceCapabilities.height = surfaceCapabilities.maxImageExtent.height;
		}
	}

	else
	{
		// If the surface size is defined, the swap chain size must match
		extentFromSurfaceCapabilities.width = surfaceCapabilities.currentExtent.width;
		extentFromSurfaceCapabilities.height = surfaceCapabilities.currentExtent.height;
	}
	return extentFromSurfaceCapabilities;
}

//Creates a Vulkan surface
vk::SurfaceKHR createVulkanSurface(const vk::Instance& instance, SDL_Window* window)
{
    SDL_SysWMinfo windowInfo;
    SDL_VERSION(&windowInfo.version);
    if(!SDL_GetWindowWMInfo(window, &windowInfo)) {
        throw std::system_error(std::error_code(), "SDK window manager info is not available.");
    }

    switch(windowInfo.subsystem) {

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
