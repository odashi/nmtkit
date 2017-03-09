# SYNOPSIS
#
#   AX_DYNET()
#
# DESCRIPTION
#
#   Sets compiler/linker flags for the DyNet library.
#
#   This macro calls:
#
#     AC_SUBST(DYNET_CPPFLAGS) / AC_SUBST(DYNET_LDFLAGS)

AC_DEFUN([AX_DYNET],
[
AC_REQUIRE([AX_CUDA])

AC_ARG_WITH([dynet],
  [AS_HELP_STRING([--with-dynet=DIR], [Location to the DyNet library])],
  [dynet_dir="${withval}"],
  [dynet_dir=""])

if test "x$dynet_dir" != "x"; then
  if test "x$cuda_dir" != "x"; then
    # DyNet with CUDA.
    DYNET_CPPFLAGS="-I${dynet_dir}/include"
    DYNET_LDFLAGS="-L${dynet_dir}/lib -lgdynet"
  else
    # DyNet with CPU.
    DYNET_CPPFLAGS="-I${dynet_dir}/include"
    DYNET_LDFLAGS="-L${dynet_dir}/lib -ldynet"
  fi
else
  AS_ERROR(Must specify --with-dynet=DIR)
fi

AC_SUBST(DYNET_CPPFLAGS)
AC_SUBST(DYNET_LDFLAGS)
])
