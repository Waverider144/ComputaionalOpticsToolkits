cmake_minimum_required(VERSION 3.10)
project(DiffraPID)

# 1. 查找依赖
find_package(Torch REQUIRED)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)


# 2. 包含头文件目录
include_directories(include)
include_directories(src)

# 3. 收集源文件
set(SOURCES
    src/main.cpp
    src/ASM.cpp
    src/DiffEngine.cpp
    src/PIDController.cpp
    src/Annealer.cpp
    src/ImageLoader.cpp
)

# 4. 生成可执行文件
add_executable(diffra_pid ${SOURCES})
target_link_libraries(diffra_pid "${TORCH_LIBRARIES}" "${OpenCV_LIBS}")

# 解决 Windows 下的编译配置（如果需要）
if (MSVC)
  file(COPY ${TORCH_INSTALL_PREFIX}/lib DESTINATION ${CMAKE_BINARY_DIR})
endif()

# 自动将配置文件拷贝到编译目录
configure_file(${CMAKE_SOURCE_DIR}/config.json ${CMAKE_BINARY_DIR}/config.json COPYONLY)