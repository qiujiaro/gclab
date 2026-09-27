add_library(testlib_main STATIC testlib_main.cc)
target_include_directories(testlib_main PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(testlib_main PUBLIC gclab_core)

# One binary per Module-1 step so you can make them pass one at a time.
foreach(t
    test_step1_cost_model
    test_step2_channel
    test_step3_transport
    test_step4_collectives)
  add_executable(${t} ${t}.cc)
  target_link_libraries(${t} PRIVATE testlib_main)
  add_test(NAME ${t} COMMAND ${t})
  # Every test is bounded: a hang becomes a failure, never a stuck terminal.
  set_tests_properties(${t} PROPERTIES TIMEOUT 60 LABELS "cpu_only;module1")
endforeach()
