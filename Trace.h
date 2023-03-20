#ifndef INTERDEVCOPY_TRACE_H_
#define INTERDEVCOPY_TRACE_H_
#include <config.h>

#ifdef ENABLE_DTRACE
#include "probes.h"
#define INTERDEVCOPY_TRACE(probe) probe
#define INTERDEVCOPY_TRACE_ENABLED(probe) probe ## _ENABLED()
#else
#define INTERDEVCOPY_TRACE(probe)
#define INTERDEVCOPY_TRACE_ENABLED(probe) (0)
#endif

#endif
