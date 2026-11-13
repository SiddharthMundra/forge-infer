cmake_minimum_required(VERSION 3.20)

# Detect CUDA before project() so we can enable the language cleanly.
include(CheckLanguage)
check_language(CUDA)

set(_llm_cuda_default OFF)
if(CMAKE_CUDA_COMPILER)
  set(_llm_cuda_default ON)
endif()

option(LLM_ENABLE_CUDA "Build CUDA kernels" ${_llm_cuda_default})
option(LLM_ENABLE_GRPC "Build gRPC server" ON)

# RTX 2070 = Turing = sm_75. Override with -DLLM_CUDA_ARCHS=75
set(LLM_CUDA_ARCHS "75" CACHE STRING "CUDA architectures (e.g. 75 for RTX 2070)")

if(LLM_ENABLE_CUDA)
  project(llm_inference_engine LANGUAGES CXX CUDA)
else()
  project(llm_inference_engine LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(MSVC)
  add_compile_options(/W3 /permissive- /Zc:__cplusplus)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
else()
  add_compile_options(-Wall -Wextra)
endif()

# Homebrew (macOS) + common Windows vcpkg prefixes.
list(APPEND CMAKE_PREFIX_PATH
  /opt/homebrew
  /opt/homebrew/opt/protobuf
  /opt/homebrew/opt/grpc
)
if(DEFINED ENV{VCPKG_ROOT})
  list(APPEND CMAKE_PREFIX_PATH "$ENV{VCPKG_ROOT}/installed/x64-windows")
endif()

find_package(Threads REQUIRED)

set(LLM_ENGINE_SOURCES
  src/engine/block_manager.cpp
  src/engine/kv_cache.cpp
  src/engine/scheduler.cpp
  src/engine/inference_engine.cpp
  src/model/weights.cpp
  src/model/tokenizer.cpp
  src/model/sampler.cpp
)

if(LLM_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  list(APPEND LLM_ENGINE_SOURCES src/kernels/ops_cuda.cu)
  add_compile_definitions(LLM_HAS_CUDA=1)
  message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")
  message(STATUS "CUDA architectures: ${LLM_CUDA_ARCHS}")
else()
  list(APPEND LLM_ENGINE_SOURCES src/kernels/ops_cpu.cpp)
endif()

add_library(llm_engine ${LLM_ENGINE_SOURCES})
target_include_directories(llm_engine PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(llm_engine PUBLIC Threads::Threads)

if(LLM_ENABLE_CUDA)
  target_link_libraries(llm_engine PUBLIC CUDA::cudart)
  set_target_properties(llm_engine PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "${LLM_CUDA_ARCHS}"
  )
  if(MSVC)
    # MSVC host compiler flags for .cu files.
    target_compile_options(llm_engine PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4819>
    )
  endif()
endif()

if(LLM_ENABLE_GRPC)
  find_package(Protobuf CONFIG)
  if(NOT Protobuf_FOUND)
    find_package(Protobuf REQUIRED)
  endif()
  find_package(gRPC CONFIG REQUIRED)

  set(PROTO_FILE ${CMAKE_SOURCE_DIR}/proto/inference.proto)
  set(PROTO_GEN_DIR ${CMAKE_BINARY_DIR}/generated)
  file(MAKE_DIRECTORY ${PROTO_GEN_DIR})

  set(PROTO_SRCS ${PROTO_GEN_DIR}/inference.pb.cc)
  set(PROTO_HDRS ${PROTO_GEN_DIR}/inference.pb.h)
  set(GRPC_SRCS ${PROTO_GEN_DIR}/inference.grpc.pb.cc)
  set(GRPC_HDRS ${PROTO_GEN_DIR}/inference.grpc.pb.h)

  get_target_property(GRPC_CPP_PLUGIN_LOCATION gRPC::grpc_cpp_plugin LOCATION)
  if(NOT GRPC_CPP_PLUGIN_LOCATION)
    find_program(GRPC_CPP_PLUGIN_LOCATION grpc_cpp_plugin REQUIRED)
  endif()

  if(TARGET protobuf::protoc)
    set(PROTOC_EXE $<TARGET_FILE:protobuf::protoc>)
  else()
    find_program(PROTOC_EXE protoc REQUIRED)
  endif()

  # On Windows the plugin path may contain spaces; pass via --plugin.
  add_custom_command(
    OUTPUT ${PROTO_SRCS} ${PROTO_HDRS} ${GRPC_SRCS} ${GRPC_HDRS}
    COMMAND ${PROTOC_EXE}
    ARGS
      --proto_path=${CMAKE_SOURCE_DIR}/proto
      --cpp_out=${PROTO_GEN_DIR}
      --grpc_out=${PROTO_GEN_DIR}
      --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN_LOCATION}
      ${PROTO_FILE}
    DEPENDS ${PROTO_FILE}
    COMMENT "Generating gRPC sources from inference.proto"
    VERBATIM
  )

  add_library(llm_proto ${PROTO_SRCS} ${GRPC_SRCS})
  target_include_directories(llm_proto PUBLIC ${PROTO_GEN_DIR})
  target_link_libraries(llm_proto PUBLIC protobuf::libprotobuf gRPC::grpc++)

  add_library(llm_server src/server/grpc_server.cpp)
  target_include_directories(llm_server PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${PROTO_GEN_DIR}
  )
  target_link_libraries(llm_server PUBLIC llm_engine llm_proto gRPC::grpc++)

  add_executable(llm_server_app apps/server_main.cpp)
  target_link_libraries(llm_server_app PRIVATE llm_server)
endif()

add_executable(llm_bench apps/bench_local.cpp)
target_link_libraries(llm_bench PRIVATE llm_engine)

message(STATUS "LLM_ENABLE_CUDA=${LLM_ENABLE_CUDA}")
message(STATUS "LLM_ENABLE_GRPC=${LLM_ENABLE_GRPC}")
