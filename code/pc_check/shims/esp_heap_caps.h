/* PC 验证用 shim: heap_caps_malloc 直接退化为 malloc */
#pragma once
#include <stdlib.h>
#define MALLOC_CAP_SPIRAM 1
#define MALLOC_CAP_8BIT   2
static inline void *heap_caps_malloc(size_t n, int caps) { (void)caps; return malloc(n); }
