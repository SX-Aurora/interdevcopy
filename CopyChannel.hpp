/**
 * @file CopyChannel.hpp
 * @brief Copy channel header
 */
#ifndef INTERDEVCOPY_COPY_CHANNEL_HPP_INCLUDE_
#define INTERDEVCOPY_COPY_CHANNEL_HPP_INCLUDE_

#include "Common.hpp"

#include <functional>
#include <typeinfo>
#include <typeindex>
#include <tuple>

namespace interdevcopy {

class DeviceMemoryRegion;
/**
 * the type of copy function
 */
using CopyFuncType = std::function<ssize_t(size_t, size_t, size_t, void *)>;
/**
 * the type of copy function getter
 */
using CopyFuncGetterType = std::function<CopyFuncType(
    DeviceMemoryRegion *, DeviceMemoryRegion *, void *)>;

/**
 * Copy Channel
 */
class CopyChannel: public util::Object {
private:
  /// source device memory region
  DeviceMemoryRegion *source;
  /// destination device memory region
  DeviceMemoryRegion *destination;
  CopyFuncType copyfunc;
public:
  CopyChannel(DeviceMemoryRegion *, DeviceMemoryRegion *, void *);
  ~CopyChannel();
  ssize_t doCopy(size_t dstoff, size_t srcoff, size_t size, void *option);
};

void _registerCopyFuncGetterMapEntry_(std::type_index, std::type_index,
    CopyFuncGetterType);

/**
 * @brief template class for registering CopyFuncGetter type
 * @tparam FuncGetter class to define CopyFuncGetter
 *
 * The template class _CopyFuncGetterMapEntryInitializer registers
 * a CopyFuncGetter on initialization.
 *
 * CopyFuncGetter is a class with the definitions of two alias types
 * and operator() function:
 *   - srctype, the type of source device memory region,
 *   - dsttype, the type of destination device memory region, and
 *   - operator() to return copy function, a function object of
 *     CopyFuncType, with the source and destination device memory region
 *     being bound.
 *
 * This template structure is intended to be created as a global object
 * in the REGISTER_COPY_HANDLER_IF macro.
 */
template <class FuncGetter> struct _CopyFuncGetterMapEntryInitializer {
  FuncGetter getter_;
  /**
   * @brief constructor to run a routine on initialization
   * @param cond condition expression
   *
   * Register CopyFuncGetter if the condition cond is true.
   */
  _CopyFuncGetterMapEntryInitializer(std::function<bool()> &&cond) {
    if (cond()) {
      _registerCopyFuncGetterMapEntry_(
          typeid(typename FuncGetter::desttype),
          typeid(typename FuncGetter::srctype), this->getter_);
    }
  }
};

/**
 * @brief a utility function to downcast and bind device memory regions
 * @tparam DstDevMemType destination device memory region type
 * @tparam SrcDevMemType source device memory region type
 * @param fp function pointer to copy data
 * @param d_ destination device memory region
 * @param s_ source device memory region
 *
 * The template function wrapCopyFuncWithDownCastAndBind() binds
 * source and destination device memory regions to a function to copy data
 * and return the lambda object with CopyFunc type.
 * This template function is used for convenience of implementation of
 * CopyFunctions (for interdevcopy) using simple memcpy-like copy functions.
 */
template <class DstDevMemType, class SrcDevMemType>
CopyFuncType wrapCopyFuncWithDownCastAndBind(
    ssize_t(*fp)(DstDevMemType *, SrcDevMemType *, size_t, size_t, size_t,
      void *), DeviceMemoryRegion *d_, DeviceMemoryRegion *s_) {
  INTERDEVCOPY_ASSERT(typeid(*d_) == typeid(DstDevMemType));
  INTERDEVCOPY_ASSERT(typeid(*s_) == typeid(SrcDevMemType));
  auto d_impl = dynamic_cast<DstDevMemType *>(d_);
  auto s_impl = dynamic_cast<SrcDevMemType *>(s_);
  return [fp, d_impl, s_impl](size_t d_off, size_t s_off, size_t size,
      void *opt) {
    return (*fp)(d_impl, s_impl, d_off, s_off, size, opt);
  };
}
} // namespace interdevcopy

/**
 * @ingroup API
 * @brief copy channel (exported data type as part of API)
 * @see interdevcopy_create_channel
 */
struct interdevcopy_channel {
  interdevcopy::CopyChannel impl;
};
/**
 * @var interdevcopy_channel::impl
 * @brief created copy channel
 */


/**
 * @brief register a CopyFuncGetter under a condition
 * @param FuncGetterType_ type name of CopyFuncGetter class
 * @param condexpr_ a condition expression
 */
#define REGISTER_COPY_HANDLER_IF(FuncGetterType_, condexpr_) \
namespace { \
  interdevcopy::_CopyFuncGetterMapEntryInitializer<FuncGetterType_> \
      _INTERDEVCOPY_UNIQUE_SYMBOL__(_copy_func_getter_init_)( \
          [](){return (condexpr_);}); \
}

/**
 * @brief register a CopyFuncGetter always
 * @param FuncGetterType_ type name of CopyFuncGetter class
 */
#define REGISTER_COPY_HANDLER(FuncGetterType_) \
  REGISTER_COPY_HANDLER_IF(FuncGetterType_, true)

#endif
