# SYNOPSIS
#
#   AX_EIGEN()
#
# DESCRIPTION
#
#   Sets compiler flags for the Eigen library.
#
#   This macro calls:
#
#     AC_SUBST(EIGEN_CPPFLAGS)

AC_DEFUN([AX_EIGEN],
[
AC_ARG_WITH([eigen],
  [AS_HELP_STRING([--with-eigen=DIR], [Location to Eigen library])],
  [eigen_dir="${withval}"],
  [eigen_dir=""])

if test "x$eigen_dir" != "x"; then
  EIGEN_CPPFLAGS="-I${eigen_dir}"
else
  AS_ERROR(Must specify --with-eigen=DIR)
fi

AC_SUBST(EIGEN_CPPFLAGS)
])
