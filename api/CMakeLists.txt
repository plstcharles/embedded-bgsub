
add_library(embedded_bgsub_api STATIC)

target_sources(
    embedded_bgsub_api
        PRIVATE
            "src/api.cpp"
        PUBLIC
            "include/api.hpp"
)

target_include_directories(
    embedded_bgsub_api
        PUBLIC
            "${CMAKE_SOURCE_DIR}/api/include"
)
