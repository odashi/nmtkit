# SYNOPSIS
#
#   AX_CUDA()
#
# DESCRIPTION
#
#   Sets compiler flags for the CUDA library.
#   This macro does not check whether the CUDA library is available or not for
#   now.
#
#   This macro calls:
#
#     AC_SUBST(CUDA_CPPFLAGS) / AC_SUBST(CUDA_LDFLAGS)
#
#   And sets:
#
#     HAVE_CUDA

AC_DEFUN([AX_CUDA],
[
AC_ARG_WITH([cuda],
  [AS_HELP_STRING([--with-cuda=DIR], [Location to the CUDA library])],
  [cuda_dir="${withval}"],
  [cuda_dir=""])

if test "x$cuda_dir" != "x"; then
  CUDA_CPPFLAGS="-I${cuda_dir}/include"
  CUDA_LDFLAGS="-L${cuda_dir}/lib64 -lcublas -lcudart"
  AC_DEFINE(HAVE_CUDA,1,[defined as 1 if the --with-cuda option is set])
else
  CUDA_CPPFLAGS=""
  CUDA_LDFLAGS=""
  AC_DEFINE(HAVE_CUDA,0,[defined as 1 if the --with-cuda option is set])
fi

AC_SUBST(CUDA_CPPFLAGS)
AC_SUBST(CUDA_LDFLAGS)
])
