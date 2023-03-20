lib_LTLIBRARIES = libinterdevcopy.la
include_HEADERS = interdevcopy.h
if INSTALL_LDSO_CONF
ldsoconf_DATA = interdevcopy.conf
endif
CLEANFILES = interdevcopy.spec interdevcopy.conf
EXTRA_DIST = COPYING README.md libinterdevcopy.map \
             interdevcopy.spec interdevcopy.spec.in \
             libinterdevcopy.stp.in

libinterdevcopy_la_SOURCES = \
    Assert.hpp Assert.cpp Common.hpp Demangle.cpp \
    DeviceMemoryRegion.cpp DeviceMemoryRegion.hpp \
    CopyChannel.cpp CopyChannel.hpp \
    CudaWrapper.cpp CudaWrapper.hpp VEOWrapper.cpp VEOWrapper.hpp \
    HostMemoryRegion.hpp HostMemory.cpp CUDAMemoryRegion.hpp CUDAMemory.cpp \
    VEOHmemRegion.hpp VEOHmem.cpp \
    CopyGPUVE.cpp \
    Trace.h probes.d

if ENABLE_DTRACE
BUILT_SOURCES = probes.h
tapset_DATA = libinterdevcopy.stp
endif

libinterdevcopy_la_LDFLAGS = -export-dynamic \
    $(GCOV_OPT) -Wl,--version-script=$(srcdir)/libinterdevcopy.map \
    -version-info $(LIB_VERSION_INFO)

edit = sed -e 's|@PACKAGE[@]|$(PACKAGE)|g' \
           -e 's|@VERSION[@]|$(VERSION)|g' \
           -e 's|@RELEASE[@]|$(RELEASE)|g' \
           -e 's|@prefix[@]|$(prefix)|g' \
           -e 's|@includedir[@]|$(includedir)|g' \
           -e 's|@libdir[@]|$(libdir)|g' \
           -e 's|@veo_includedir[@]|$(veo_includedir)|g' \
           -e 's|@veo_libdir[@]|$(veo_libdir)|g' \
           -e 's|@cuda_includedir[@]|$(cuda_includedir)|g' \
           -e 's|@cuda_libdir[@]|$(cuda_libdir)|g' \
           -e 's|@veo_prefix[@]|$(veo_prefix)|g' \
           -e 's|@cuda_prefix[@]|$(cuda_prefix)|g'

interdevcopy.spec: $(srcdir)/interdevcopy.spec.in
	$(edit) $< > $@

if INSTALL_LDSO_CONF
interdevcopy.conf:
	echo $(libdir) > $@
endif

if ENABLE_DTRACE
probes.h: $(srcdir)/probes.d
	$(DTRACECMD) -C -h -s $< -o $@
probes.o: $(srcdir)/probes.d
	$(DTRACECMD) -C -G -s $< -o $@
libinterdevcopy.stp: $(srcdir)/libinterdevcopy.stp.in
	$(edit) $< > $@

libinterdevcopy_la_LIBADD = probes.o
endif

rpm: dist
	rpmbuild -ta --clean ${distdir}.tar.gz
