! mf_test.f90 — Fortran msk144decodeframe phase correction + matched filter + bpdecode128_90.
!
! Reads n_frames complex64 frames (each 864 complex samples) from stdin in text format.
! Runs phase correction + matched filter + ssig normalize + LLR compute + bpdecode128_90.
! Outputs one line per frame: nsuccess nharderror
! Does NOT require packjt77 (message unpack is skipped; success = CRC pass).

program mf_test
  implicit none

  integer, parameter :: NSPM = 864, N = 128, M = 38, K = 90
  real, parameter :: sigma = 0.60
  integer, parameter :: max_iterations = 10

  complex :: c(NSPM), cb(42), cfac, cca, ccb
  real :: pp(12), softbits(144), llr(N), sav, s2av, ssig
  integer*1 :: s8(8), hardbits(144)
  integer*1 :: message77(77), apmask(N), cw(N)
  integer :: nbadsync, nbadsync1, nbadsync2, nharderror, niterations
  integer :: i, nframes, ifrm, nsuccess
  real :: phase0, cbi_arr(42), cbq_arr(42)
  double precision :: pi
  integer :: ios

  ! Build sync template (from msk144decodeframe.f90)
  pi = 4d0 * datan(1d0)
  do i = 1, 12
    pp(i) = sin((i-1) * pi / 12.0)
  enddo
  s8 = (/0,1,1,1,0,0,1,0/)
  s8 = 2*s8 - 1
  cbq_arr(1:6) = pp(7:12) * s8(1)
  cbq_arr(7:18) = pp * s8(3)
  cbq_arr(19:30) = pp * s8(5)
  cbq_arr(31:42) = pp * s8(7)
  cbi_arr(1:12) = pp * s8(2)
  cbi_arr(13:24) = pp * s8(4)
  cbi_arr(25:36) = pp * s8(6)
  cbi_arr(37:42) = pp(1:6) * s8(8)
  cb = cmplx(cbi_arr, cbq_arr)

  apmask = 0

  read(*, *, iostat=ios) nframes
  if (ios /= 0 .or. nframes <= 0) then
    write(*,'(a)') 'ERROR: nframes'
    stop 1
  endif

  do ifrm = 1, nframes
    ! Read 864 complex values as 1728 whitespace-separated floats (re im re im ...)
    block
      real :: buf(2*NSPM)
      read(*, *, iostat=ios) buf
      if (ios /= 0) then
        write(*,'(a,i0)') 'ERROR reading frame ', ifrm
        stop 1
      endif
      do i = 1, NSPM
        c(i) = cmplx(buf(2*i-1), buf(2*i))
      enddo
    end block

    ! Phase correction
    cca = sum(c(1:42) * conjg(cb))
    ccb = sum(c(337:378) * conjg(cb))
    phase0 = atan2(imag(cca + ccb), real(cca + ccb))
    cfac = cmplx(cos(phase0), sin(phase0))
    c = c * conjg(cfac)

    ! Matched filter
    softbits(1) = sum(imag(c(1:6))*pp(7:12)) + sum(imag(c(859:864))*pp(1:6))
    softbits(2) = sum(real(c(1:12))*pp)
    do i = 2, 72
      softbits(2*i-1) = sum(imag(c(1+(i-1)*12-6 : 1+(i-1)*12+5)) * pp)
      softbits(2*i)   = sum(real(c(7+(i-1)*12-6 : 7+(i-1)*12+5)) * pp)
    enddo

    ! Sync gate
    hardbits = 0
    where (softbits >= 0.0) hardbits = 1
    nbadsync1 = (8 - sum((2*hardbits(1:8)-1) * s8)) / 2
    nbadsync2 = (8 - sum((2*hardbits(57:64)-1) * s8)) / 2
    nbadsync = nbadsync1 + nbadsync2
    if (nbadsync > 4) then
      write(*,'(a)') '0 -1 0'
      call flush(6)
      cycle
    endif

    ! Normalize
    sav  = sum(softbits) / 144.0
    s2av = sum(softbits * softbits) / 144.0
    ssig = sqrt(max(s2av - sav*sav, 1e-10))
    softbits = softbits / ssig

    ! LLR
    llr(1:48)   = softbits(9:56)
    llr(49:128) = softbits(65:144)
    llr = 2.0 * llr / (sigma * sigma)

    ! LDPC decode
    message77 = 0
    nharderror = -1
    niterations = 0
    call bpdecode128_90(llr, apmask, max_iterations, message77, cw, nharderror, niterations)

    nsuccess = 0
    if (nharderror >= 0 .and. nharderror < 18) nsuccess = 1

    write(*, '(3i6)') nsuccess, nharderror, niterations
    call flush(6)
  enddo

end program mf_test
