#ifndef _CODECS_BUILD_CPP_INCLUDE_HPP
#define _CODECS_BUILD_CPP_INCLUDE_HPP

#include <cstdlib>
#include <new>

extern "C" inline void __cxa_pure_virtual() { std::abort(); }
extern "C" inline void* __cxa_allocate_exception(size_t) throw() { std::abort(); }
extern "C" inline void __cxa_throw() { std::abort(); }
extern "C" inline unsigned int __cxa_uncaught_exceptions() { return 0; }

inline void* operator new(std::size_t n)
#if __cplusplus < 201703L
    throw(std::bad_alloc)
#endif
{
    void* ret = std::malloc(n);
    if (ret == nullptr) {
        std::abort();
    }
    return ret;
}

inline void* operator new[](std::size_t n)
#if __cplusplus < 201703L
    throw(std::bad_alloc)
#endif
{
    void* ret = std::malloc(n);
    if (ret == nullptr) {
        std::abort();
    }
    return ret;
}

inline void* operator new(std::size_t n, std::align_val_t a)
#if __cplusplus < 201703L
    throw(std::bad_alloc)
#endif
{
    void* ret = aligned_alloc(static_cast<std::size_t>(a), n);
    if (ret == nullptr) {
        std::abort();
    }
    return ret;
}

inline void operator delete(void* p) noexcept
{
    std::free(p);
}

inline void operator delete[](void* p) noexcept
{
    std::free(p);
}

inline void operator delete(void* p, std::align_val_t a) noexcept
{
    std::free(p);
}

inline void operator delete(void* p, std::size_t sz) noexcept
{
    std::free(p);
}

#endif // _CODECS_BUILD_CPP_INCLUDE_HPP
