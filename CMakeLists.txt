
# Set the minimum CMake version required to build the project.
cmake_minimum_required( VERSION 3.1 )

# Silence some warnings on macOS with new CMake versions.
if( NOT ${CMAKE_VERSION} VERSION_LESS 3.9 )
   cmake_policy( SET CMP0068 NEW )
endif()

# Set the project's name and version.
project( lwtnn VERSION 2.10 )

# Enable using CTest in the project.
include( CTest )

# Set up the "C++ version" to use.
set( CMAKE_CXX_STANDARD_REQUIRED 11 CACHE STRING
   "Minimum C++ standard required for the build" )
set( CMAKE_CXX_STANDARD 11 CACHE STRING
   "C++ standard to use for the build" )
set( CMAKE_CXX_EXTENSIONS FALSE CACHE BOOL
   "(Dis)allow using compiler extensions" )

# If the user didn't request a build type explicitly, use an optimised
# build with debug symbols.
if( "${CMAKE_BUILD_TYPE}" STREQUAL "" )
   set( CMAKE_BUILD_TYPE "RelWithDebInfo" )
endif()
message( STATUS "Using build type: ${CMAKE_BUILD_TYPE}" )

# Set the warning flag(s) to use.
set( CMAKE_CXX_FLAGS "-Wall -pedantic -Wextra -Wcomment -Wunused-value" )

# Turn off the usage of RPATH completely:
set( CMAKE_SKIP_RPATH ON )
set( CMAKE_SKIP_BUILD_RPATH ON )
set( CMAKE_SKIP_INSTALL_RPATH ON )

# Set the location of the built libraries/executables inside the build
# directory.
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin )
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib )
set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib )

# Make the project's modules visible to CMake.
list( INSERT CMAKE_MODULE_PATH 0 ${CMAKE_SOURCE_DIR}/cmake )

# Figure out where to take "externals" from. Either using externals already
# available on the build system, or downloading the necessary code as part
# of the build of this project.

# Figure out what to do with Boost.
option( BUILTIN_BOOST "Acquire Boost as part of building this project" OFF )
if( BUILTIN_BOOST )
   # Download and install the Boost headers using ExternalProject. Note that
   # we don't need any of the Boost libraries, so we're not using Boost's own
   # build system. And we only need the Boost headers during the build
   # "privately", so the headers are not installed with the project.
   include( ExternalProject )
   ExternalProject_Add( Boost
      PREFIX ${CMAKE_BINARY_DIR}/externals
      INSTALL_DIR ${CMAKE_BINARY_DIR}/externals/Boost
      URL
      "https://dl.bintray.com/boostorg/release/1.64.0/source/boost_1_64_0.tar.gz"
      URL_MD5 "319c6ffbbeccc366f14bb68767a6db79"
      BUILD_IN_SOURCE 1
      CONFIGURE_COMMAND ${CMAKE_COMMAND} -E echo "No configuration for Boost"
      BUILD_COMMAND ${CMAKE_COMMAND} -E echo "No build step for Boost"
      INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/boost
      <INSTALL_DIR>/include/boost )
   # Set the include path to use.
   set( Boost_INCLUDE_DIRS
      $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/externals/Boost/include> )
   # Tell the user what will happen.
   message( STATUS "Using a privately downloaded Boost version for the build" )
else()
   # Look for Boost on the build system.
   find_package( Boost 1.54.0 REQUIRED )
endif()

# Figure out what to do with Eigen.
option( BUILTIN_EIGEN "Acquire Eigen as part of building this project" OFF )
if( BUILTIN_EIGEN )
   # Download and install Eigen using ExternalProject.
   include( ExternalProject )
   ExternalProject_Add( Eigen
      PREFIX ${CMAKE_BINARY_DIR}/externals
      INSTALL_DIR ${CMAKE_BINARY_DIR}/externals/Eigen
      URL "https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.bz2"
      URL_MD5 "b9e98a200d2455f06db9c661c5610496"
      DOWNLOAD_NAME "eigen-3.3.7.tar.bz2"
      CMAKE_CACHE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR> )
   # The eigen headers are needed by clients of this project as well, so let's
   # install them with the project.
   install( DIRECTORY ${CMAKE_BINARY_DIR}/externals/Eigen/
      DESTINATION . USE_SOURCE_PERMISSIONS )
   # Set the include path to use.
   set( EIGEN3_INCLUDE_DIR
      $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/externals/Eigen/include/eigen3>
      $<INSTALL_INTERFACE:include/eigen3> )
   # Tell the user what will happen.
   message( STATUS "Using a privately downloaded Eigen version for the build" )
else()
   # Look for Eigen on the build system.
   find_package( Eigen3 REQUIRED )
endif()

# Public header files for the shared/static library.
set( lib_headers
   include/lwtnn/generic/eigen_typedefs.hh
   include/lwtnn/generic/Graph.hh
   include/lwtnn/generic/Graph.tcc
   include/lwtnn/generic/InputPreprocessor.hh
   include/lwtnn/generic/InputPreprocessor.tcc
   include/lwtnn/generic/LightweightNeuralNetwork.hh
   include/lwtnn/generic/LightweightNeuralNetwork.tcc
   include/lwtnn/generic/Stack.hh
   include/lwtnn/generic/Stack.tcc
   include/lwtnn/generic/Source.hh
   include/lwtnn/generic/LightweightGraph.hh
   include/lwtnn/generic/LightweightGraph.tcc
   include/lwtnn/generic/FastInputPreprocessor.hh
   include/lwtnn/generic/FastInputPreprocessor.tcc
   include/lwtnn/generic/FastGraph.hh
   include/lwtnn/generic/FastGraph.tcc
   include/lwtnn/Exceptions.hh
   include/lwtnn/LightweightNeuralNetwork.hh
   include/lwtnn/NNLayerConfig.hh
   include/lwtnn/NanReplacer.hh
   include/lwtnn/lightweight_network_config.hh
   include/lwtnn/lightweight_nn_streamers.hh
   include/lwtnn/parse_json.hh )
# Source files for the shared/static library.
set( lib_sources
   src/Exceptions.cxx
   src/LightweightNeuralNetwork.cxx
   src/LightweightGraph.cxx
   src/NanReplacer.cxx
   src/FastInputPreprocessor.cxx
   src/FastGraph.cxx
   src/Stack.cxx
   src/lightweight_nn_streamers.cxx
   src/parse_json.cxx
   src/test_utilities.hh
   src/test_utilities.cxx )

# Build the shared library.
add_library( lwtnn SHARED ${lib_headers} ${lib_sources} )
target_include_directories( lwtnn
   PUBLIC ${EIGEN3_INCLUDE_DIR}
   $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>
   PRIVATE ${Boost_INCLUDE_DIRS} )
set_property( TARGET lwtnn
   PROPERTY PUBLIC_HEADER ${lib_headers} )
if( BUILTIN_BOOST )
   add_dependencies( lwtnn Boost )
endif()
if( BUILTIN_EIGEN )
   add_dependencies( lwtnn Eigen )
endif()

# Build the static library.
add_library( lwtnn-stat ${lib_headers} ${lib_sources} )
target_include_directories( lwtnn-stat
   PUBLIC ${EIGEN3_INCLUDE_DIR}
   $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>
   PRIVATE ${Boost_INCLUDE_DIRS} )
set_property( TARGET lwtnn-stat
   PROPERTY PUBLIC_HEADER ${lib_headers} )
if( BUILTIN_BOOST )
   add_dependencies( lwtnn-stat Boost )
endif()
if( BUILTIN_EIGEN )
   add_dependencies( lwtnn-stat Eigen )
endif()

# Install the libraries.
install( TARGETS lwtnn lwtnn-stat
   EXPORT lwtnnTargets
   ARCHIVE DESTINATION lib
   LIBRARY DESTINATION lib
   PUBLIC_HEADER DESTINATION include/lwtnn )

# Helper macro for building the projects' executables.
macro( lwtnn_add_executable name )
   # Declare the executable.
   add_executable( ${name} src/${name}.cxx )
   # Set its properties.
   target_link_libraries( ${name} lwtnn-stat )
   target_include_directories( ${name} SYSTEM PRIVATE ${Boost_INCLUDE_DIRS} )
   # Install it.
   install( TARGETS ${name}
      EXPORT lwtnnTargets
      RUNTIME DESTINATION bin )
endmacro( lwtnn_add_executable )

# Build/install all executables of the project.
lwtnn_add_executable( lwtnn-benchmark-rnn )
lwtnn_add_executable( lwtnn-dump-config )
lwtnn_add_executable( lwtnn-test-arbitrary-net )
lwtnn_add_executable( lwtnn-test-graph )
lwtnn_add_executable( lwtnn-test-lightweight-graph )
lwtnn_add_executable( lwtnn-test-rnn )
lwtnn_add_executable( lwtnn-test-fastgraph )

# Install the converter scripts.
install( FILES
   converters/keras_layer_converters_common.py
   converters/keras_v1_layer_converters.py
   converters/keras_v2_layer_converters.py
   DESTINATION converters )
install( PROGRAMS
   converters/keras2json.py
   converters/sequential2graph.py
   converters/kerasfunc2json.py
   converters/sklearn2json.py
   DESTINATION converters )

# Helper macro for building the project's unit tests.
macro( lwtnn_add_test name )
   # Build the unit-test executable:
   add_executable( ${name} ${ARGN} )
   target_link_libraries( ${name} lwtnn-stat )
   set_target_properties( ${name} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test-bin" )
   # Set up the test itself:
   add_test( NAME ${name}_ctest
      COMMAND ${CMAKE_BINARY_DIR}/test-bin/${name}
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/test-bin )
endmacro( lwtnn_add_test )

# Set up the test(s) of the project.
lwtnn_add_test( test-nn-streamers tests/test-nn-streamers.cxx )

# Install the CMake description of the project.
install( EXPORT lwtnnTargets
   FILE lwtnnConfig-targets.cmake
   DESTINATION cmake
   NAMESPACE "lwtnn::" )
configure_file( ${CMAKE_SOURCE_DIR}/cmake/lwtnnConfig-version.cmake.in
   ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/lwtnnConfig-version.cmake @ONLY )
install( FILES ${CMAKE_SOURCE_DIR}/cmake/lwtnnConfig.cmake
   ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/lwtnnConfig-version.cmake
   DESTINATION cmake )
