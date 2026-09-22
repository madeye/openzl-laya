if(NOT COMMAND add_zs_test)
  function(add_zs_test TESTNAME)
      add_executable(${ARGV})
      target_link_libraries(${TESTNAME}
          PRIVATE
              GTest::gtest_main
      )
      gtest_discover_tests(${TESTNAME})
  endfunction()
endif()

enable_testing()

if (OPENZL_BUILD_TESTS)
    add_zs_test(cli_util_test unittest/UtilTest.cpp)
    target_link_libraries(cli_util_test PRIVATE cli_util)
    target_include_directories(cli_util_test PRIVATE
        ${PROJECT_SOURCE_DIR})
    apply_openzl_compile_options_to_target(cli_util_test)

    find_package(Python3 COMPONENTS Interpreter)
    if (Python3_FOUND)
        add_test(
            NAME cli_integration_tests
            COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/cli_integration_tests.py
                $<TARGET_FILE:zli>
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
    endif()
endif()

if(OPENZL_BUILD_TESTS)
    add_executable(laya_fixture laya_fixture.cpp)
    target_link_libraries(laya_fixture PRIVATE commands)
    target_include_directories(laya_fixture PRIVATE ${PROJECT_SOURCE_DIR})
    apply_openzl_compile_options_to_target(laya_fixture)
    add_zs_test(laya_test unittest/LayaTest.cpp)
    target_link_libraries(laya_test PRIVATE commands)
    target_include_directories(laya_test PRIVATE ${PROJECT_SOURCE_DIR})
    apply_openzl_compile_options_to_target(laya_test)
    if(OPENZL_ENABLE_LAYA AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
        # Tokenizer and model parity of the native CUDA worker against fixtures
        # produced by the upstream PyTorch implementation. Skips (exit 77)
        # when the pinned assets have not been prepared.
        add_test(NAME laya_native_check
            COMMAND openzl-laya-worker check
                ${CMAKE_CURRENT_SOURCE_DIR}/laya_reference.json
                ${CMAKE_CURRENT_SOURCE_DIR}/laya_tokenizer_reference.json)
        set_tests_properties(laya_native_check PROPERTIES SKIP_RETURN_CODE 77 LABELS laya)
    endif()
endif()
