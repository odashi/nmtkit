# nmtkit unit tests

# test definitions
function(nmtkit_test name)
  add_executable(${name}_test ${name}_test.cc)
  target_link_libraries(${name}_test nmtkit ${Boost_LIBRARIES})
  add_test(
    NAME ${name}_test
    COMMAND ${name}_test
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
endfunction()

nmtkit_test(array)
nmtkit_test(batch_converter)
nmtkit_test(bpe_vocabulary)
nmtkit_test(character_vocabulary)
nmtkit_test(corpus)
nmtkit_test(inference_graph)
nmtkit_test(monotone_sampler)
nmtkit_test(random)
nmtkit_test(sorted_random_sampler)
nmtkit_test(unicode)
nmtkit_test(word_vocabulary)
