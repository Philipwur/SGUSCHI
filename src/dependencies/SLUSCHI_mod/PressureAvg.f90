program DetermineSize
implicit none

  integer i,j
  integer nwork
  integer ios
  double precision vol
  double precision,dimension(6)::row
  double precision,allocatable,dimension(:,:)::pressure3
  double precision,dimension(6)::pressure3_avg !Pxx Pyy Pzz Pxy Pyz Pzx
  double precision pressure_kinetic,pressure_Pulay,pressure_target
  double precision,dimension(6)::pressure3_total !Pxx Pyy Pzz Pxy Pyz Pzx

  ! read pressure3.out; average
  call OpenInput(7,'pressure3.out')
  nwork=0
  do while (.true.)
    read(7,*,iostat=ios) row
    if (ios < 0) exit
    if (ios > 0) call Fatal('invalid numeric row in pressure3.out')
    do j=1,6
      if (.not. IsFinite(row(j))) call Fatal('non-finite value in pressure3.out')
    enddo
    nwork = nwork+1
  enddo
  if (nwork < 1) call Fatal('pressure3.out contains no valid rows')

  rewind 7
  allocate(pressure3(nwork,6),stat=ios)
  if (ios /= 0) call Fatal('could not allocate pressure array')
  do i=1,nwork
    read(7,*,iostat=ios) pressure3(i,1:6)
    if (ios /= 0) call Fatal('failed while rereading pressure3.out')
  enddo
  close(7)

  pressure3_avg = 0.d0
  do i=1,nwork
    pressure3_avg = pressure3_avg + pressure3(i,1:6)
  enddo
  pressure3_avg = pressure3_avg/dble(nwork)

  call OpenInput(8,'volume.out')
  nwork=0
  do while (.true.)
    read(8,*,iostat=ios) vol
    if (ios < 0) exit
    if (ios > 0) call Fatal('invalid numeric value in volume.out')
    if (.not. IsFinite(vol)) call Fatal('non-finite value in volume.out')
    nwork = nwork+1
  enddo
  close(8)
  if (nwork < 1) call Fatal('volume.out contains no valid rows')
  if (vol <= 0.d0) call Fatal('last volume.out value must be positive')
  pressure3_avg = pressure3_avg*1602.177d0/vol

  ! read kinetic pressure, Pulay stress, and target pressure; calculate total pressure
  call ReadScalar('pressure_kinetic.out',pressure_kinetic)
  call ReadScalar('pressure_Pulay.out',pressure_Pulay)
  call ReadScalar('pressure_target.out',pressure_target)

  pressure3_total = pressure3_avg
  pressure3_total(1:3) = pressure3_total(1:3) + pressure_kinetic + pressure_Pulay - pressure_target

  ! output pressure3_total
  open(11,file='pressure3_total.out',iostat=ios)
  if (ios /= 0) call Fatal('could not open pressure3_total.out')
  write(11,"(6F15.6)") pressure3_total
  close(11)

contains

  subroutine OpenInput(Unit,FileName)
    integer,intent(in)::Unit
    character(len=*),intent(in)::FileName

    open(Unit,file=FileName,status='old',action='read',iostat=ios)
    if (ios /= 0) call Fatal('could not open '//trim(FileName))
  end subroutine OpenInput

  subroutine ReadScalar(FileName,Value)
    character(len=*),intent(in)::FileName
    double precision,intent(out)::Value
    integer unit

    unit = 8
    if (FileName == 'pressure_Pulay.out') unit = 9
    if (FileName == 'pressure_target.out') unit = 10

    call OpenInput(unit,FileName)
    read(unit,*,iostat=ios) Value
    if (ios /= 0) call Fatal('invalid or missing value in '//trim(FileName))
    close(unit)

    if (.not. IsFinite(Value)) call Fatal('non-finite value in '//trim(FileName))
  end subroutine ReadScalar

  logical function IsFinite(Value)
    double precision,intent(in)::Value

    IsFinite = (Value == Value) .and. (dabs(Value) <= huge(Value))
  end function IsFinite

  subroutine Fatal(Message)
    character(len=*),intent(in)::Message

    write(*,'(A)') 'FATAL PressureAvg.x: '//trim(Message)
    stop 1
  end subroutine Fatal

end
