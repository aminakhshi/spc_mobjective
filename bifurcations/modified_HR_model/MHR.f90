!---------------------------------------------------------------------- 
!---------------------------------------------------------------------- 
!   Modified HR model for bursting patterns
!---------------------------------------------------------------------- 
!----------------------------------------------------------------------
!

SUBROUTINE FUNC(NDIM,U,ICP,PAR,IJAC,F,DFDU,DFDP)
!--------- ---- 

    IMPLICIT NONE
    INTEGER, INTENT(IN) :: NDIM, IJAC, ICP(*)
    DOUBLE PRECISION, INTENT(IN) :: U(NDIM), PAR(*)
    DOUBLE PRECISION, INTENT(OUT) :: F(NDIM), DFDU(NDIM,*), DFDP(NDIM,*)
    
    ! Constants
    DOUBLE PRECISION, PARAMETER :: a=0.61123046875, b=3.430957031249, c=1.7927734375, d=5.737792968
    DOUBLE PRECISION, PARAMETER :: r=0.0050932617, s=3.373046875, v_R=-1.6667187
    DOUBLE PRECISION, PARAMETER :: g_u=0.9, g_d=0.05, a_z=0.2
    DOUBLE PRECISION, PARAMETER :: c_n=0.6, x_v=0.5, k_n=0.066666667
    DOUBLE PRECISION, PARAMETER :: c_h=0.3, x_z=0.5, k_h=0.133333334
    
    ! Variables
    DOUBLE PRECISION :: vs, ys, zs, us
    ! Equations
    DOUBLE PRECISION :: ninf_v, hinf_z
    ! Parameters
    DOUBLE PRECISION :: Iapp
    
    ! Definition of variables
    vs = U(1)
    ys = U(2)
    zs = U(3)
    us = U(4)

    ! Definition of parameters
    Iapp = PAR(1)
    
    ! Equations of the system
    ninf_v = c_n * (1 + tanh((vs - x_v)/k_n))
    hinf_z = c_h * (1 + tanh((zs - x_z)/k_h))

    F(1) = ys - a * (vs**3) + b * (vs**2) - zs - us + Iapp
    F(2) = c - d * (vs**2) - ys
    F(3) = r * (s * (vs - v_R) - zs)
    F(4) = - g_u * ninf_v + a_z * hinf_z - g_d * us 

END SUBROUTINE FUNC

!---------------------------------------------------------------------- 
SUBROUTINE STPNT(NDIM,U,PAR,T)
!--------- ---- 

    IMPLICIT NONE
    INTEGER, INTENT(IN) :: NDIM
    DOUBLE PRECISION, INTENT(IN) :: T
    DOUBLE PRECISION, INTENT(INOUT) :: U(NDIM),PAR(*)
    ! Initial condition 
    U(1:4) = (/ -3.089821, -52.985906, -4.8001512, 0.0 /)
    ! System parameters
    PAR(1) = -2.6

END SUBROUTINE STPNT

!---------------------------------------------------------------------- 
SUBROUTINE PVLS(NDIM,U,PAR)
!   ---------------
    IMPLICIT NONE
    INTEGER, INTENT(IN) :: NDIM
    DOUBLE PRECISION, INTENT(IN) :: U(NDIM)
    DOUBLE PRECISION, INTENT(INOUT) :: PAR(*)

    DOUBLE PRECISION, EXTERNAL :: GETP
    INTEGER NDX,NCOL,NTST

    ! Set PAR(9) equal to the minimum of U(1)
    PAR(9)=GETP('MIN',1,U)

END SUBROUTINE PVLS

!---------------------------------------------------------------------- 
! The following subroutines are not used here,
! but they must be supplied as dummy routines

SUBROUTINE BCND 
END SUBROUTINE BCND

SUBROUTINE ICND 
END SUBROUTINE ICND

SUBROUTINE FOPT 
END SUBROUTINE FOPT
