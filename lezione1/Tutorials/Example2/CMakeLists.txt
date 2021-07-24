cmake_minimum_required (VERSION 2.8.11)
project (IFOA2021_example_cpp_thread)

add_executable (esempioThread esempioThread.cpp)

target_link_libraries (esempioThread -lpthread)