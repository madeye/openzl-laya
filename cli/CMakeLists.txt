# Copyright (c) Meta Platforms, Inc. and affiliates.

file(GLOB cli_root_sources
    CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

file(GLOB cli_root_headers
    CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.h")

add_subdirectory(args)
add_subdirectory(commands)
add_subdirectory(utils)

if (OPENZL_BUILD_TESTS)
    add_subdirectory(tests)
endif()

add_executable(zli
    ${cli_root_sources}
)
target_link_libraries(zli PRIVATE
    args
    commands
    utils
    logger
)

# there's gotta be a better way than manually specifying openzl/include/
target_include_directories(zli
    PRIVATE ${PROJECT_BINARY_DIR}/include
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR} )

add_dependencies(zli
    args
    commands
    utils
    logger
)

if(TARGET xgboost_external)
    add_dependencies(zli xgboost_external)
endif()

apply_openzl_compile_options_to_target(zli)

if(OPENZL_ENABLE_LAYA)
    target_compile_definitions(args PUBLIC OPENZL_ENABLE_LAYA=1)
    target_compile_definitions(utils PUBLIC OPENZL_ENABLE_LAYA=1)
    # One C++ worker for both platforms; only the model backend differs.
    set(LAYA_WORKER_SOURCES laya/worker.cpp laya/tokenizer.cpp)
    if(APPLE)
        if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
            message(FATAL_ERROR "Laya requires Apple Silicon and macOS 14+")
        endif()
        execute_process(COMMAND sw_vers -productVersion OUTPUT_VARIABLE LAYA_MACOS_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(LAYA_MACOS_VERSION VERSION_LESS 14 OR (CMAKE_OSX_DEPLOYMENT_TARGET AND CMAKE_OSX_DEPLOYMENT_TARGET VERSION_LESS 14))
            message(FATAL_ERROR "Laya requires macOS 14+")
        endif()
        # Core ML backend: a thin Objective-C++ file over the pinned .mlmodelc.
        enable_language(OBJCXX)
        add_executable(openzl-laya-worker ${LAYA_WORKER_SOURCES} laya/coreml/model.mm)
        target_compile_options(openzl-laya-worker PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
        target_link_libraries(openzl-laya-worker PRIVATE "-framework CoreML" "-framework Foundation")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        # Native CUDA backend; `openzl-laya-worker prepare` downloads the
        # pinned checkpoint.
        if(NOT CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES native)
        endif()
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)
        find_package(Threads REQUIRED)
        add_executable(openzl-laya-worker ${LAYA_WORKER_SOURCES} laya/cuda/model.cu)
        target_compile_features(openzl-laya-worker PRIVATE cuda_std_17)
        target_link_libraries(openzl-laya-worker PRIVATE CUDA::cudart CUDA::cublasLt CUDA::cublas Threads::Threads)
    else()
        message(FATAL_ERROR "Laya requires macOS 14+ on Apple Silicon or Linux with CUDA")
    endif()
    target_include_directories(openzl-laya-worker PRIVATE ${PROJECT_SOURCE_DIR})
    target_compile_features(openzl-laya-worker PRIVATE cxx_std_17)
    set_target_properties(openzl-laya-worker PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
    add_dependencies(zli openzl-laya-worker)
    if(OPENZL_INSTALL)
        install(TARGETS openzl-laya-worker RUNTIME DESTINATION ${OPENZL_INSTALL_BINDIR})
    endif()
endif()
if(OPENZL_INSTALL)
    install(TARGETS zli RUNTIME DESTINATION ${OPENZL_INSTALL_BINDIR})
endif()
