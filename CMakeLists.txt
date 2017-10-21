project(UnscentedKF)

set(CMAKE_BUILD_TYPE Debug)

cmake_minimum_required (VERSION 3.5)

add_definitions(-std=c++0x)

set(sources
   src/ukf.cpp
   src/main.cpp
   src/tools.cpp)

set(testsources
   src/ukf.cpp
   src/testmain.cpp
   src/tools.cpp)
   
add_executable(UnscentedKF ${sources})

add_executable(testUnscentedKF ${testsources})
