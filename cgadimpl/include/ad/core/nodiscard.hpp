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
