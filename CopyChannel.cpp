/**
 * @file CopyChannel.cpp
 * @brief Implementation of CopyChannel
 */

#include "CopyChannel.hpp"
#include "DeviceMemoryRegion.hpp"
#include "Trace.h"
#include <unordered_map>

namespace {
using interdevcopy::CopyFuncGetterType;
/**
 * Hold mappings from source and destination device region type to
 * CopyFuncGetter, a factory of function object for data transfer.
 *  i.e. getter_map: type_index->(type_index->CopyFuncGetterType).
 */
std::unordered_map<std::type_index,
  std::unordered_map<std::type_index, CopyFuncGetterType>> getter_map;
} // unnamed

namespace interdevcopy {

/**
 * @brief Register a CopyFuncGetter
 * @param dsttype destination device memory region type as type_index
 * @param srctype source device memory region type as type_index
 *
 * The function _registerCopyFuncGetterMapEntry_() registers
 * a CopyFuncGeter, a factory class with operator() to create a
 * copy function (closure) from source and destination device memory region
 * type and an option.
 * This function is intended to be called from a constructor of
 * global _CopyFuncGetterMapEntryInitializer (template class)
 * object for initialization.
 */
void _registerCopyFuncGetterMapEntry_(std::type_index dsttype,
    std::type_index srctype, CopyFuncGetterType getter) {
  auto elem = getter_map.find(dsttype);
  if (elem != getter_map.end()) {
    // found
    auto &map_from_srctype = elem->second;
    INTERDEVCOPY_ASSERT(map_from_srctype.find(srctype)
        == map_from_srctype.end());
    map_from_srctype[srctype] = getter;
  } else {
    // not found; create newmap=(srctype -> getter), and
    // add newmap to getter_map (dsttype -> newmap)
    auto newmap = decltype(getter_map)::mapped_type{{srctype, getter}};
    getter_map[dsttype] = newmap;
  }
  if (INTERDEVCOPY_TRACE_ENABLED(LIBINTERDEVCOPY_CHANNEL_REGISTER_TYPE)) {
    auto srcname = util::demangle(srctype.name());
    auto dstname = util::demangle(dsttype.name());
    auto gettername = util::demangle(typeid(getter).name());
    INTERDEVCOPY_TRACE(LIBINTERDEVCOPY_CHANNEL_REGISTER_TYPE(dstname.c_str(),
          srcname.c_str(), gettername.c_str()));
  }
}

/**
 * @brief Get a copy function object for copy channel.
 * @param dst_ destination device memory region
 * @param src_ source device memory region
 * @param option for future use
 * @return a function object to copy from source to destination
 *
 * Find a CopyFuncGetter for source and destination device memory region
 * type, create a function object and binds source and destination device
 * memory region to the function object, using the CopyFuncGetter.
 * This function is called from the constructor of CopyChannel.
 * to initialize the member copyfunc.
 */
CopyFuncType getCopyFunc(DeviceMemoryRegion *dst_,
    DeviceMemoryRegion *src_, void *option) {
  auto elem = getter_map.find(typeid(*dst_));
  if (elem == getter_map.end()) {
    auto dstname = util::demangle(typeid(*dst_).name());
    INTERDEVCOPY_TRACE(LIBINTERDEVCOPY_CHANNEL_DESTINATION_TYPE_NOT_FOUND(
          dstname.c_str()));
    throw std::system_error(std::make_error_code(std::errc::not_supported),
          "Unsupported destination device memory type");
  }
  auto map_from_srctype = elem->second;
  auto copyfunc_elem = map_from_srctype.find(typeid(*src_));
  if (copyfunc_elem == map_from_srctype.end()) {
    auto dstname = util::demangle(typeid(*dst_).name());
    auto srcname = util::demangle(typeid(*src_).name());
    INTERDEVCOPY_TRACE(LIBINTERDEVCOPY_CHANNEL_SOURCE_TYPE_NOT_FOUND(
          dstname.c_str(), srcname.c_str()));
    throw std::system_error(std::make_error_code(std::errc::not_supported),
          "Unsupported source device memory type");
  }
  auto getter = copyfunc_elem->second;
  if (INTERDEVCOPY_TRACE_ENABLED(LIBINTERDEVCOPY_CHANNEL_GETTER_FOUND)) {
    auto dstname = util::demangle(typeid(*dst_).name());
    auto srcname = util::demangle(typeid(*src_).name());
    auto gettertype = util::demangle(typeid(getter).name());
    INTERDEVCOPY_TRACE(LIBINTERDEVCOPY_CHANNEL_GETTER_FOUND(
          dstname.c_str(), srcname.c_str(), gettertype.c_str()));
  }
  return getter(dst_, src_, option);
}

/**
 * @brief constructor
 * @param dst destination device memory region
 * @param src source device memory region
 * @param option for future use
 *
 * Constructor of Copy Channel: set source, destination and copyfunc.
 */
CopyChannel::CopyChannel(DeviceMemoryRegion *dst, DeviceMemoryRegion *src,
    void *option):
  source(src), destination(dst), copyfunc(getCopyFunc(dst, src, option)) {
  // All members including copyfunc have been successfully initialized here,
  // hence, increment the usage counts.
  dst->get();
  src->get();
}

/**
 * @brief destructor
 */
CopyChannel::~CopyChannel() {
  this->source->put();
  this->destination->put();
}

/**
 * @brief copy data
 * @param dstoff offset of destination
 * @param srcoff offset of source
 * @param size copy size in byte
 * @param option for future use
 * @return the size of copied data upon success;
 *         an exception is thrown upon failure.
 *
 * Copy data from source device memory region to destination device memory
 * region using copyfunc of the copy channel.
 */
ssize_t CopyChannel::doCopy(size_t dstoff, size_t srcoff, size_t size,
    void *option) {
  this->get();
  util::Finally f([this](){this->put();});
  // check range
  // source
  if (srcoff >= this->source->getSize()) {
    throw std::invalid_argument("source offset is out of range.");
  }
  if (srcoff + size > this->source->getSize()) {
    throw std::invalid_argument("source size is out of range.");
  }
  // destination
  if (dstoff >= this->destination->getSize()) {
    throw std::invalid_argument("destination offset is out of range.");
  }
  if (dstoff + size > this->destination->getSize()) {
    throw std::invalid_argument("destination size is out of range.");
  }
  return this->copyfunc(dstoff, srcoff, size, option);
}

/**
 * @brief implementation of interdevcopy_create_channel().
 * @param dst destination device memory region
 * @param src source device memory region
 * @param option for future use
 * @return pointer to interdevcopy_channel upon sucess;
 *         an exception is thrown upon failure.
 *
 * This function implements an interdevcopy API function
 * interdevcopy_create_channel().
 */
interdevcopy_channel *createChannel_(DeviceMemoryRegion *dst,
    DeviceMemoryRegion *src, void *option) {
  auto rv = new interdevcopy_channel{.impl = {dst, src, option}};
  INTERDEVCOPY_TRACE(LIBINTERDEVCOPY_CHANNEL_CREATE(dst, src, rv));
  return rv;
}

/**
 * @brief implementation of interdevcopy_delete_channel()
 * @param ch pointer to interdevcopy_channel
 * @return zero upon success; an exception is thrown upon failure.
 *
 * This function implements an interdevcopy API function
 * interdevcopy_delete_channel().
 */
int deleteChannel_(interdevcopy_channel *ch) {
  if (ch == nullptr) {
    std::string msg("invalid memory pointer ");
    throw std::invalid_argument(msg);
  }
  if (ch->impl.isUsed()) {
    throw std::system_error(std::make_error_code(
          std::errc::device_or_resource_busy));
  }
  delete ch;
  return 0;
}

} // namespace interdevcopy

/* API functions */
/**
 * @ingroup API
 * @brief API function to create a copy channel.
 * @param dst destination device memory region
 * @param src source device memory region
 * @param option for future use
 * @return pointer to copy channel upon success; negative upon failure.
 */
interdevcopy_channel *interdevcopy_create_channel(
    interdevcopy_memory_region *dst, interdevcopy_memory_region *src,
    void *option) {
  if (src == nullptr || dst == nullptr) {
    return interdevcopy::util::Cast<interdevcopy_channel *>(-EINVAL);
  }
  INTERDEVCOPY_API_WRAPPER(interdevcopy::createChannel_(dst->implp,
        src->implp, option));
}

/**
 * @ingroup API
 * @brief API function to destoy a copy channel
 * @param ch a copy channel
 * @return zero upon success; negative upon failure.
 */
int interdevcopy_destroy_channel(interdevcopy_channel *ch) {
  INTERDEVCOPY_API_WRAPPER(interdevcopy::deleteChannel_(ch));
}

/**
 * @ingroup API
 * @brief API function to perform data copy
 * @param ch copy channel
 * @param dstoff offset of destination
 * @param srcoff offset of source
 * @param size copy size in byte
 * @param option for future use
 * @return data the size of copied data upon success; negative upon failure.
 */
ssize_t interdevcopy_copy(interdevcopy_channel *ch, unsigned long dstoff,
    unsigned long srcoff, size_t size, void *option) {
  if (ch == nullptr) {
    return interdevcopy::util::Cast<ssize_t>(-EINVAL);
  }
  INTERDEVCOPY_API_WRAPPER(ch->impl.doCopy(dstoff, srcoff, size, option));
}
