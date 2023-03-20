AC_PREREQ([2.69])
AC_INIT([interdevcopy], [0.1.0b1], [dev-nlcpy@sxarr.jp.nec.com])
# Libtool library versioning information current:release:age 
# https://www.gnu.org/software/libtool/manual/libtool.html#Versioning
LIB_VERSION_INFO='1:0:0'
AM_INIT_AUTOMAKE([foreign -Wall -Werror])
RELEASE="1"
AC_SUBST(LIB_VERSION_INFO)
AC_SUBST(RELEASE)

AC_CONFIG_MACRO_DIR([m4])
AC_CONFIG_HEADER([config.h])

# options for RELEASE VERSION
AC_ARG_WITH([release], [AS_HELP_STRING([--with-release],
  [Specify release version])],
  [RELEASE="${with_release}"])
AC_SUBST(RELEASE)

# options for CUDA
AC_ARG_WITH([cuda], [AS_HELP_STRING([--with-cuda],
  [CUDA installation path [/usr/local/cuda]])],
  [cuda_prefix="${with_cuda}"], [cuda_prefix=/usr/local/cuda])
AC_SUBST(cuda_prefix)
AC_ARG_WITH([cuda-includedir], [AS_HELP_STRING([--with-cuda-includedir],
  [CUDA header path [cuda/include]])],
  [cuda_includedir="${with_cuda_includedir}"],
  [cuda_includedir="${cuda_prefix}/include"])
AC_SUBST(cuda_includedir)
AC_ARG_WITH([cuda-libdir], [AS_HELP_STRING([--with-cuda-libdir],
  [CUDA library path [cuda/lib64]])],
  [cuda_libdir="${with_cuda_libdir}"], [cuda_libdir="${cuda_prefix}/lib64"])
AC_SUBST(cuda_libdir)

# options for VEO
AC_ARG_WITH([veo], [AS_HELP_STRING([--with-veo],
  [VE Offload installation path [/opt/nec/ve/veos]])],
  [veo_prefix="${with_veo}"], [veo_prefix=/opt/nec/ve/veos])
AC_SUBST(veo_prefix)
AC_ARG_WITH([veo-includedir], [AS_HELP_STRING([--with-veo-includedir],
  [VE offload header path [veo/include]])],
  [veo_includedir="${with_veo_includedir}"],
  [veo_includedir="${veo_prefix}/include"])
AC_SUBST(veo_includedir)
AC_ARG_WITH([veo-libdir], [AS_HELP_STRING([--with-veo-libdir],
  [VE offload library path [veo/lib64]])],
  [veo_libdir="${with_veo_libdir}"], [veo_libdir="${veo_prefix}/lib64"])
AC_SUBST(veo_libdir)

# options for ld.so.conf
AC_ARG_WITH([ldso-conf], [AS_HELP_STRING([--with-ldso-conf],
  [ld.so configuration path [/etc/ld.so.conf.d]])],
  [], [with_ldso_conf=/etc/ld.so.conf.d])
AS_IF([test "x$with_ldso_conf" != xno],
  [ldsoconfdir="$with_ldso_conf"
   AC_SUBST(ldsoconfdir)
  ], [])
AM_CONDITIONAL(INSTALL_LDSO_CONF, [test "x$with_ldso_conf" != xno])

# support dtrace
AC_ARG_ENABLE([dtrace], [AS_HELP_STRING([--enable-dtrace],
  [Enable DTrace/SystemTap support.])], [ENABLE_DTRACE="$enableval"],
  [ENABLE_DTRACE="no"])
AM_CONDITIONAL([ENABLE_DTRACE], [test x"$ENABLE_DTRACE" = xyes])
AC_CHECK_HEADERS([sys/sdt.h])
AC_CHECK_PROGS(DTRACECMD, dtrace)
if test "x${ENABLE_DTRACE}" = xyes && test "x${ac_cv_header_sys_sdt_h}" != xyes
then
  AC_MSG_ERROR([DTrace support requires <sys/sdt.h>. Install the header.])
fi
if test "x${ENABLE_DTRACE}" = xyes && test x"${DTRACECMD}" = x
then
  AC_MSG_ERROR([Dtrace support requires dtrace command. Install the command.])
fi
if test x"${ENABLE_DTRACE}" = xyes; then
  AC_DEFINE(ENABLE_DTRACE, [],
    [Define to 1 if DTrace/SystemTap tracing is enabled.])
fi
AC_ARG_WITH([tapsetdir], [AS_HELP_STRING([--with-tapsetdir],
  [Tapset installation path [datadir/systemtap/tapset]])],
  [tapsetdir="${with_tapsetdir}"], [tapsetdir="${datadir}/systemtap/tapset"])
AC_SUBST(tapsetdir)

###
# Unit test option
AC_ARG_ENABLE(coverage,
[  --enable-coverage      enable coverage test, only for developer of this package. Please "Don't" use this package if you install it, gcov is GPL license [[default=no]]],
[\
case "${enableval}" in
 yes) enable_coverage=yes ;;
 *)   AC_MSG_ERROR(bad value for --enable-) ;;
esac],
enable_coverage=no)

#check flag, and add gcov option
if test x"${enable_coverage}" = x"yes"; then
  AC_CHECK_LIB([gcov], [main])
  CPPFLAGS="$CPPFLAGS -coverage"
  GCOV_OPT="-coverage"
  AC_SUBST([GCOV_OPT])
fi

# Checks for programs.
AC_PROG_CXX
AM_PROG_AR
LT_INIT

# Checks for libraries.
AC_CHECK_LIB([dl], [dlopen], [], [AC_MSG_ERROR([libdl library is missing.])])

# Checks for header files.
CPPFLAGS="$CPPFLAGS -I${veo_includedir}"
AC_CHECK_HEADERS([ve_offload.h veo_dev_mem.h veo_vedma.h])
CPPFLAGS="$CPPFLAGS -I${cuda_includedir}"
AC_CHECK_HEADERS([cuda.h])
AC_LANG(C++)
AC_CHECK_HEADERS([cxxabi.h])

# Checks for typedefs, structures, and compiler characteristics.

# Checks for library functions.

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
