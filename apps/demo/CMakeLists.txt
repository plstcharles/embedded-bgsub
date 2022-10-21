
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(
    embedded_bgsub_demo
        "src/demo_main.cpp"
)

target_include_directories(
    embedded_bgsub_demo
        PUBLIC
            "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/api/include>"
)

target_link_libraries(
    embedded_bgsub_demo
        PUBLIC
            "${OpenCV_LIBS}"
            embedded_bgsub_api
)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_link_libraries(  # for <execution> stdlib
        embedded_bgsub_demo
            PRIVATE
                TBB::tbb
    )
endif ()

set_target_properties(
    embedded_bgsub_demo
        PROPERTIES
            FOLDER "apps"
)

install(
    TARGETS embedded_bgsub_demo
    RUNTIME DESTINATION "bin"
    COMPONENT "apps"
)
