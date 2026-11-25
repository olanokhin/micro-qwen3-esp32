# The same microqwen.c that runs on the Mac (firmware/microqwen.c). On the ESP32-S3: PIE vector kernel (MQ_PIE),
# aligned buffers in internal SRAM (MQ_ESP32). Web demo: Wi-Fi access point + HTTP + SSE,
# the index.html page is embedded in the firmware.
idf_component_register(SRCS "main.c" "../microqwen.c"
                       INCLUDE_DIRS "." ".."
                       EMBED_TXTFILES "index.html"
                       PRIV_REQUIRES esp_partition esp_timer driver esp_driver_usb_serial_jtag vfs spi_flash heap
                                     esp_wifi esp_http_server esp_netif esp_event nvs_flash)
target_compile_options(${COMPONENT_LIB} PRIVATE -O2 -Wno-unused-function -ffp-contract=off)
target_compile_definitions(${COMPONENT_LIB} PRIVATE MQ_ESP32=1 MQ_PIE=1)
