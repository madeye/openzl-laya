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
    if(NOT APPLE OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
        message(FATAL_ERROR "Laya requires Apple Silicon and macOS 14+")
    endif()
    execute_process(COMMAND sw_vers -productVersion OUTPUT_VARIABLE LAYA_MACOS_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(LAYA_MACOS_VERSION VERSION_LESS 14 OR (CMAKE_OSX_DEPLOYMENT_TARGET AND CMAKE_OSX_DEPLOYMENT_TARGET VERSION_LESS 14))
        message(FATAL_ERROR "Laya requires macOS 14+")
    endif()
    find_program(SWIFT_EXECUTABLE swift REQUIRED)
    target_compile_definitions(args PUBLIC OPENZL_ENABLE_LAYA=1)
    target_compile_definitions(utils PUBLIC OPENZL_ENABLE_LAYA=1)
    file(GLOB_RECURSE LAYA_SWIFT_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/laya/Sources/*.swift")
    add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/openzl-laya-worker"
        COMMAND ${SWIFT_EXECUTABLE} build --package-path "${CMAKE_CURRENT_SOURCE_DIR}/laya" --scratch-path "${CMAKE_CURRENT_SOURCE_DIR}/laya/.build" -c release --disable-automatic-resolution
        COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/laya/.build/release/openzl-laya-worker" "${CMAKE_CURRENT_BINARY_DIR}/openzl-laya-worker"
        DEPENDS ${LAYA_SWIFT_SOURCES} laya/Package.swift laya/Package.resolved
        VERBATIM)
    add_custom_target(laya_worker ALL DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/openzl-laya-worker")
    add_dependencies(zli laya_worker)
    if(OPENZL_INSTALL)
        install(PROGRAMS "${CMAKE_CURRENT_BINARY_DIR}/openzl-laya-worker" DESTINATION ${OPENZL_INSTALL_BINDIR})
    endif()
endif()
if(OPENZL_INSTALL)
    install(TARGETS zli RUNTIME DESTINATION ${OPENZL_INSTALL_BINDIR})
endif()
