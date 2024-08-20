#include <fstream>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <vulkan/vulkan.h>

const std::vector<const char*> g_validationLayers = {
    "VK_LAYER_KHRONOS_validation",
};

static uint32_t FindQueueFamily(const VkPhysicalDevice device, bool *hasIdx);
static uint32_t FindMemoryType(const VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties);
static std::vector<uint32_t> LoadSPIRV(const std::string name);

struct VulkanAbstract
{
    VkInstance instance;
    VkPhysicalDevice physicalDevice;
    VkDevice device;
    VkQueue queue;
    VkShaderModule computeShader;
    VkDescriptorSetLayout descriptorSetLayout;
    VkPipelineLayout computePipelineLayout;
    VkPipeline computePipeline;
    VkDescriptorPool descriptorPool;
    VkDescriptorSet descriporSet;
    VkCommandPool cmdPool;
    VkCommandBuffer cmdBuffer;
    VkFence fence;

    uint32_t graphicsQueueFamilyIdx;
    bool enableValidationLayers = ((getenv("DEMO_USE_VALIDATION") != NULL) && (strncmp("1", getenv("DEMO_USE_VALIDATION"), 2) == 0));


    void InitVulkan()
    {
        {
            std::vector<const char*> extensions{};

            // 1.1. Specify the application infromation.
            // One important info is the "apiVersion"
            VkApplicationInfo appInfo;
            {
                appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
                appInfo.pNext = NULL;
                appInfo.pApplicationName = "MinimalVkcompute";
                appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
                appInfo.pEngineName = "RAW";
                appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
                appInfo.apiVersion = VK_API_VERSION_1_0;
            }

            // 1.2. Specify the Instance creation information.
            // The Instance level Validation and debug layers must be specified here.
            VkInstanceCreateInfo createInfo;
            {
                createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
                createInfo.pNext = NULL;
                createInfo.flags = 0;
                createInfo.pApplicationInfo = &appInfo;
                createInfo.enabledLayerCount = 0;

                if (enableValidationLayers) {
                    createInfo.enabledLayerCount = static_cast<uint32_t>(g_validationLayers.size());
                    createInfo.ppEnabledLayerNames = g_validationLayers.data();
                    // If a debug callbacks should be enabled:
                    //  * The extension must be specified and
                    //  * The "pNext" should point to a valid "VkDebugUtilsMessengerCreateInfoEXT" struct.
                    // extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
                    //createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*) &debugInfo;
                }

                createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
                createInfo.ppEnabledExtensionNames = extensions.data();
            }

            // 1.3. Create the Vulkan instance.
            if (vkCreateInstance(&createInfo, NULL, &instance) != VK_SUCCESS) {
                throw std::runtime_error("failed to create instance!");
            }
        }

        {
            // 2.1 Query the number of physical devices.
            uint32_t deviceCount = 0;
            vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

            if (deviceCount == 0) {
                throw std::runtime_error("failed to find GPUs with Vulkan support!");
            }

            // 2.2. Get all avaliable physical devices.
            std::vector<VkPhysicalDevice> devices(deviceCount);
            vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

            // 2.3. Select a physical device (based on some info).
            // Currently the first physical device is selected if it supports Graphics Queue.
            for (const VkPhysicalDevice& device : devices) {
                bool hasIdx;
                graphicsQueueFamilyIdx = FindQueueFamily(device, &hasIdx);
                if (hasIdx) {
                    physicalDevice = device;
                    break;
                }
            }

            if (physicalDevice == VK_NULL_HANDLE) {
                throw std::runtime_error("failed to find a suitable GPU!");
            }
        }

        {
            // 3.1. Build the device queue create info data (use only a singe queue).
            float queuePriority = 1.0f;
            VkDeviceQueueCreateInfo queueCreateInfo;
            {
                queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
                queueCreateInfo.pNext = NULL;
                queueCreateInfo.flags = 0;
                queueCreateInfo.queueFamilyIndex = graphicsQueueFamilyIdx;
                queueCreateInfo.queueCount = 1;
                queueCreateInfo.pQueuePriorities = &queuePriority;
            }

            // 3.2. The queue family/families must be provided to allow the device to use them.
            std::vector<uint32_t> uniqueQueueFamilies = { graphicsQueueFamilyIdx };

            // 3.3. Specify the device creation information.
            VkDeviceCreateInfo createInfo;
            {
                createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
                createInfo.pNext = NULL;
                createInfo.flags = 0;
                createInfo.queueCreateInfoCount = 1;
                createInfo.pQueueCreateInfos = &queueCreateInfo;
                createInfo.pEnabledFeatures = NULL;
                createInfo.enabledExtensionCount = 0;
                createInfo.ppEnabledExtensionNames = NULL;
                createInfo.enabledLayerCount = 0;

                if (enableValidationLayers) {
                    // To have device level validation information, the layers are added here.
                    createInfo.enabledLayerCount = static_cast<uint32_t>(g_validationLayers.size());
                    createInfo.ppEnabledLayerNames = g_validationLayers.data();
                }
            }

            // 3.4. Create the logical device.
            if (vkCreateDevice(physicalDevice, &createInfo, NULL, &device) != VK_SUCCESS) {
                throw std::runtime_error("failed to create logical device!");
            }
        }

        {
            vkGetDeviceQueue(device, graphicsQueueFamilyIdx, 0, &queue);
        }

        {
            VkCommandPoolCreateInfo poolInfo;
            {
                poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
                poolInfo.pNext = NULL;
                poolInfo.flags = 0;
                poolInfo.queueFamilyIndex = graphicsQueueFamilyIdx;
            }

            if (vkCreateCommandPool(device, &poolInfo, NULL, &cmdPool) != VK_SUCCESS) {
                throw std::runtime_error("failed to create command pool!");
            }
        }

        {
            VkCommandBufferAllocateInfo allocInfo;
            {
                allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
                allocInfo.pNext = NULL;
                allocInfo.commandPool = cmdPool;
                allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
                allocInfo.commandBufferCount = 1;
            }

            if (vkAllocateCommandBuffers(device, &allocInfo, &cmdBuffer) != VK_SUCCESS) {
                throw std::runtime_error("failed to allocate command buffers!");
        }
    }




    }
};


int main()
{
    VulkanAbstract vkAbstract;
    vkAbstract.InitVulkan();

    while (1)
    {
        ;
    }
    return 0;
}

static uint32_t FindQueueFamily(const VkPhysicalDevice device, bool *hasIdx) {
    if (hasIdx != nullptr) {
        *hasIdx = false;
    }

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

    uint32_t queueFamilyIdx = 0;
    for (const VkQueueFamilyProperties& queueFamily : queueFamilies) {
        if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            if (hasIdx != NULL) {
                *hasIdx = true;
            }

            return queueFamilyIdx;
        }

        // TODO?: check if device supports the target surface
        queueFamilyIdx++;
    }

    return UINT32_MAX;
}


uint32_t FindMemoryType(const VkPhysicalDevice physicalDevice,
                        uint32_t typeFilter,
                        VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties{};
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("failed to find suitable memory type!");
}
