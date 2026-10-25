#pragma once
#include <cstdio>

#include "vgpu_uapi.h"

namespace vrm {

inline const char *trace_csv_header() { return "ts_ns,type,chan,engine,arg,arg2"; }

inline const char *trace_type_name(uint32_t t)
{
    switch (t) {
    case VGPU_EV_SLICE_BEGIN: return "slice_begin";
    case VGPU_EV_SLICE_END:   return "slice_end";
    case VGPU_EV_CMD:         return "cmd";
    case VGPU_EV_DOORBELL:    return "doorbell";
    case VGPU_EV_FAULT:       return "fault";
    case VGPU_EV_BLOCKED:     return "blocked";
    default:                  return "unknown";
    }
}

inline void trace_csv_row(FILE *f, const vgpu_trace_ev &e)
{
    std::fprintf(f, "%llu,%s,%u,%u,%u,%llu\n", (unsigned long long)e.ts_ns,
                 trace_type_name(e.type), e.chan, e.engine, e.arg,
                 (unsigned long long)e.arg2);
}

}  // namespace vrm
