#ifndef __ASSEMBLER__
#ifndef _CODECS_BUILD_CPP_INCLUDE_HPP
#define _CODECS_BUILD_CPP_INCLUDE_HPP

#include <cstdlib>

extern "C" inline void *__cxa_allocate_exception(size_t) throw()
{
    std::abort();
}

#endif // _CODECS_BUILD_CPP_INCLUDE_HPP
#endif // __ASSEMBLER__
