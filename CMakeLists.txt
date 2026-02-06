cmake_minimum_required(VERSION 3.16)
project(ECMC_CB)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)

include(FetchContent)
FetchContent_Declare(
        lime
        GIT_REPOSITORY https://github.com/usqcd-software/c-lime.git
        GIT_TAG        master  # Ou un tag spécifique comme v1.3.2
)

if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 not found. Download via FetchContent...")

    FetchContent_Declare(
            eigen
            GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
            GIT_TAG 3.4.0
    )
    FetchContent_MakeAvailable(eigen)
else()
    message(STATUS "Eigen3 found on the system.")
endif()

add_executable(gauge_ecmc_cb src/bin/gauge_ecmc_cb.cpp
        src/gauge/GaugeField.cpp
        src/su3/utils.cpp
        src/geometry/GeometryCB.cpp
        src/geometry/types.h
        src/mpi/HalosCB.h
        src/mpi/HalosShift.h
        src/mpi/HalosExchange.cpp
        src/mpi/MpiTopology.cpp
        src/mpi/types.h
        src/mpi/HalosObs.h
        src/observables/observables_mpi.cpp
)


# Configuration des options selon le compilateur détecté
if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    # Options spécifiques à icpx (OneAPI)
    target_compile_options(gauge_ecmc_cb PRIVATE
            -O2
            #-xHost            # Remplace -march=native pour Intel
            -fp-model precise
            -Wall -Wextra
    )
else()
    target_compile_options(gauge_ecmc_cb PRIVATE
            -O2
            #-ffast-math
            #-march=native
            -Wall -Wextra -Wpedantic
    )
endif()

target_link_libraries(gauge_ecmc_cb PRIVATE MPI::MPI_CXX Eigen3::Eigen)

