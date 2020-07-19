#ifndef __AID_TYPES_H__
#define __AID_TYPES_H__

#if defined HAVE_STDINT_H
#  include <stdint.h>
#elif defined HAVE_INTTYPES_H
#  include <inttypes.h>
#elif defined HAVE_SYS_TYPES_H
#  include <sys/types.h>
#endif

#include <stdint.h>

typedef int16_t aid_int16_t;
typedef uint16_t aid_uint16_t;
typedef int32_t aid_int32_t;
typedef uint32_t aid_uint32_t;

#endif

