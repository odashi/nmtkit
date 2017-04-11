# nmtkit frontend executables

add_executable(train train.cc)
add_executable(decode decode.cc)

target_link_libraries(train ${Boost_LIBRARIES} nmtkit mteval)
target_link_libraries(decode ${Boost_LIBRARIES} nmtkit)
