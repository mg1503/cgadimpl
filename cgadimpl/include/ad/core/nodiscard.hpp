//===========================================
// FILE: cgadimpl/include/ad/core/nodiscard.hpp
//============================================


#pragma once


#ifndef AG_NODISCARD
#  if defined(__has_cpp_attribute)
#    if __has_cpp_attribute(nodiscard)
#      define AG_NODISCARD [[nodiscard]]
#    else
#      define AG_NODISCARD
#    endif
#  else
#    define AG_NODISCARD
#  endif
#endif

// The crash happens because the code is trying to add a bias of size 5 directly to an input of size 10, 
// instead of adding it to the result of the matrix multiplication (which would have been size 5).
// This perfectly demonstrates why the AG_NODISCARD macro is useful—it would have warned you at compile-time
// that you were ignoring the result of the matmul operation!