add_executable(lin main.c)

target_link_libraries(lin PRIVATE
  InetDialect
  MLIRCAPIIR
  MLIRCAPIRegisterEverything
)

target_include_directories(lin PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)
