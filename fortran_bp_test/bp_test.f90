! bp_test.f90 — Standalone test wrapper for bpdecode128_90.
!
! Reads LLR vectors from stdin (binary, N float32 per frame), decodes each,
! writes results to stdout: one line per frame:
!   nsuccess nharderror iterations
!
! If nsuccess==1, also writes 77 decoded bits as a second line.
!
! Build:
!   cd fortran_bp_test
!   gfortran -O2 -fPIC -o bp_test bp_test.f90 bpdecode128_90.f90 platanh.f90 \
!       chkcrc13a.f90 crc_mod.f90 ../wsjtx/crc13.cpp -lstdc++ \
!       -I../wsjtx

program bp_test
  use crc
  implicit none

  integer, parameter :: N = 128
  real    :: llr(N)
  integer*1 :: message77(77), apmask(N), cw(N)
  integer :: nharderror, niterations, maxiterations
  integer :: ios, nframes, ifrm

  ! Read number of frames from stdin
  read(*, *, iostat=ios) nframes
  if (ios /= 0 .or. nframes <= 0) then
    write(*,'(a)') 'ERROR: could not read nframes'
    stop 1
  endif

  maxiterations = 10
  apmask = 0

  do ifrm = 1, nframes
    ! Read 128 LLR values
    read(*, *, iostat=ios) llr
    if (ios /= 0) then
      write(*,'(a,i0)') 'ERROR: reading frame ', ifrm
      stop 1
    endif

    message77 = 0
    nharderror = -1
    niterations = 0

    call bpdecode128_90(llr, apmask, maxiterations, message77, cw, nharderror, niterations)

    write(*, '(3i8)') merge(1, 0, nharderror >= 0 .and. nharderror < 18), &
                      nharderror, niterations
    if (nharderror >= 0 .and. nharderror < 18) then
      write(*, '(77i1)') message77
    endif
    call flush(6)
  enddo

end program bp_test
