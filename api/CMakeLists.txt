include_directories(${OpenCV_INCLUDE_DIRS})

add_library(embedded_bgsub_api STATIC)

target_sources(
    embedded_bgsub_api
        PRIVATE
            "src/BackgroundSubtractorViBe.cpp" "src/VibeBGS.cpp" "include/VibeBGSUtils.hpp" "include/vibeUtils.hpp"
        PUBLIC
            "include/BackgroundSubtractorViBe.hpp" "include/VibeBGS.hpp"
)

target_include_directories(
    embedded_bgsub_api
        PUBLIC
            "${CMAKE_SOURCE_DIR}/api/include"
)

