#ifndef RESOLVEDISM_SELFCHECK_H
#define RESOLVEDISM_SELFCHECK_H
/* Runtime invariant checks for the test harness; compiled only with STELLA_SELFCHECK.
 * Log format (parsed by stella/tests/harness): "STELLA_SELFCHECK <tag> RAN|FAIL ..." */
#include <stdio.h>

#ifdef STELLA_SELFCHECK
#define STELLA_CHECK(tag, ok, fmt, ...) do { \
    if(!(ok)) { printf("STELLA_SELFCHECK %s FAIL Task=%d " fmt "\n", tag, ThisTask, ##__VA_ARGS__); fflush(stdout); } \
} while(0)
#define STELLA_CHECK_RAN(tag, n) do { \
    if((n) > 0) { printf("STELLA_SELFCHECK %s RAN Task=%d n=%d\n", tag, ThisTask, (int)(n)); fflush(stdout); } \
} while(0)
#else
#define STELLA_CHECK(tag, ok, fmt, ...) do {} while(0)
#define STELLA_CHECK_RAN(tag, n) do {} while(0)
#endif

#endif
