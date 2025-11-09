!---------------------------------------------------------------------- 
!---------------------------------------------------------------------- 
!   Polynomial model for transient bursting patterns
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
    DOUBLE PRECISION, PARAMETER :: gnas=55.0, gdrs=20.0, gleak=0.18, gnad=5.0
    DOUBLE PRECISION, PARAMETER :: Vna=40.0, Vk=-88.5, Vleak=-70.0, Vca=100.0 
    DOUBLE PRECISION, PARAMETER :: kCa=0.4, tau_ns=0.39, tau_hd=1.0, tau_nd=0.9, tau_pd=5.0
    DOUBLE PRECISION, PARAMETER :: v_ms=40.0, v_ns=40.0, v_md=40.0, v_hd=52.0, v_nd=40.0, v_pd=65.0
    DOUBLE PRECISION, PARAMETER :: s_ms=3.0, s_ns=3.0, s_md=5.0, s_hd=5.0, s_nd=5.0, s_pd=6.0
    DOUBLE PRECISION, PARAMETER :: C=1.0, k=0.4, gc=1.0
    DOUBLE PRECISION, PARAMETER :: R_b=45.9, R_u=12.9, R_o=46.5, R_c=73.8, R_r=6.8, R_d=8.4, Mg_o=1
    DOUBLE PRECISION, PARAMETER :: IP3=0.3, nuIP3=15.0, d1=0.13, d2=1.049, d3=0.9434, d5=0.08234, a2=0.2
    DOUBLE PRECISION, PARAMETER :: nuPMCA=30.0, nuSerca=22.5, kPMCA=0.45, kSerca=0.105  
    DOUBLE PRECISION, PARAMETER :: nuINleak=0.03, nuERleak=0.03      
    DOUBLE PRECISION, PARAMETER :: fc=0.05, alpha=0.5, fER=0.025, gamma=9.0

    ! Variables
    DOUBLE PRECISION :: Vs, Vd, ns, hd, nd, pd, sd, Ca, CaE, hca
    DOUBLE PRECISION :: C0, C1, C2, Ds, Os
    ! Parameters
    DOUBLE PRECISION :: Inas, Idrs, Ileaks, Istod
    DOUBLE PRECISION :: Inad, Idrd, Ileakd, Idtos, Iskd, Inmda
    DOUBLE PRECISION :: msinf, nsinf, mdinf, hdinf, ndinf, pdinf
    DOUBLE PRECISION :: sdinf
    DOUBLE PRECISION :: Jip3, Jserca, Jpmca, JleakER
    DOUBLE PRECISION :: Q2, hinf_IP3, minf_IP3, ninf_IP3, Bv
    
    
    DOUBLE PRECISION :: Iapp, gdrd, gNMDA, gSK, tau_sd, glu
    ! Definition of variables
    Vs = U(1)
    Vd = U(2)
    ns = U(3)
    hd = U(4)
    nd = U(5)
    pd = U(6)
    sd  = U(7)
    Ca = U(8)
    CaE = U(9)
    hca = U(10)
    C0 = U(11)
    C1  = U(12)
    C2 = U(13)
    Os = U(14)

    ! Definition of parameters
    Iapp = PAR(1)
    gdrd = PAR(2)
    gNMDA = PAR(3)
    gSK  = PAR(4)
    tau_sd = PAR(5)
    glu = PAR(6)
    
    ! Equations of the system
    msinf = 1/(1+exp(-(Vs+v_ms)/s_ms))
    nsinf = 1/(1+exp(-(Vs+v_ns)/s_ns))
    mdinf = 1/(1+exp(-(Vd+v_md)/s_md))
    hdinf = 1/(1+exp((Vd+v_hd)/s_hd))
    ndinf = 1/(1+exp(-(Vd+v_nd)/s_nd))
    pdinf = 1/(1+exp((Vd+v_pd)/s_pd))
    sdinf = (0.81 * (Ca ** 4 / (Ca ** 4 + kCa ** 4)))

    Q2 = d2*((IP3+d1)/(IP3+d3))
    hinf_IP3 = Q2/(Q2+Ca)
    minf_IP3 = IP3/(IP3+d1)
    ninf_IP3 = Ca/(Ca+d5)

    ! Magnesium block B(v) for NMDA receptor; [Mg] = 1 or 2 mM
    Bv = 1/(1+(Mg_o*exp(-0.062*Vd))/3.57)

    Inas = gnas*(msinf**2)*(1-ns)*(Vs-Vna)
    Idrs = gdrs*(ns**2)*(Vs-Vk)
    Ileaks = gleak*(Vs-Vleak)
    Istod = (gc/k)*(Vs-Vd)
    
    Inad = gnad*(mdinf**2)*hd*(Vd-Vna)
    Idrd = gdrd*(nd**2)*pd*(Vd-Vk)
    Ileakd = gleak*(Vd-Vleak)
    Idtos = (gc/(1-k))*(Vd-Vs)
    Iskd = gSK*sd*(Vd-Vk)
    Inmda = gNMDA*Bv*Os*(Vd-Vca)
    
    Jip3 = nuIP3 * (minf_IP3**3) * (ninf_IP3**3) * (hCa**3) * (CaE - Ca)
    Jserca = nuSerca * (Ca**2) / (Ca**2 + kSerca**2)
    Jpmca = nuPMCA * (Ca**2) / (Ca**2 + kPMCA**2)
    JleakER = nuERleak * (CaE - Ca)
    Ds = 1 - Os - C2
    
    ! F(1) = (10**(Iapp)-Inas-Idrs-Ileaks-Istod)/C
    F(1) = (Iapp-Inas-Idrs-Ileaks-Istod)/C
    F(2) = (-Inad-Idrd-Ileakd-Idtos-Iskd-Inmda)/C
    F(3) = (nsinf-ns)/tau_ns
    F(4) = (hdinf-hd)/tau_hd
    F(5) = (ndinf-nd)/tau_nd
    F(6) = (pdinf-pd)/tau_pd
    F(7) = (sdinf-sd)/tau_sd
    F(8) = fc*(-alpha*Inmda + Jip3 - Jserca - Jpmca + JleakER)
    F(9) = fER*gamma*(-Jip3 + Jserca - JleakER)
    F(10) = (hinf_IP3 - hCa) * a2 * (Q2 + Ca)
    F(11) = -(R_b * glu * C0) + (R_u * C1)
    F(12) = -((R_b * glu + R_u) * C1) + (R_b * glu * C0 + R_u * C2)
    F(13) = -((R_o + R_d + R_u) * C2) + (R_b * glu * C1 + R_c * Os + R_r * Ds)
    F(14) = -(R_c * Os) + (R_o * C2)

 END SUBROUTINE FUNC

!---------------------------------------------------------------------- 
 SUBROUTINE STPNT(NDIM,U,PAR,T)
!--------- ---- 

    IMPLICIT NONE
    INTEGER, INTENT(IN) :: NDIM
    DOUBLE PRECISION, INTENT(IN) :: T
    !DOUBLE PRECISION, INTENT(OUT) :: U(NDIM), PAR(*)
    DOUBLE PRECISION, INTENT(INOUT) :: U(NDIM),PAR(*)
    ! Initial condition 
    U(1:14) = (/ -56.742, -57.014, 0.0037554, 0.73162, 0.032207, &
            0.209, 0.0010077, 0.075147, 22.19, 0.8284, &
            0.30629, 0.32694, 0.34899, 0.21989 /)
    ! System parameters
    PAR(1:6) = (/ 3.0, 14.0, 0.5, 0.5, 1.1, 0.3/)
 END SUBROUTINE STPNT

!---------------------------------------------------------------------- 
SUBROUTINE PVLS(NDIM,U,PAR)
!     ---------------
      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM
      DOUBLE PRECISION, INTENT(IN) :: U(NDIM)
      DOUBLE PRECISION, INTENT(INOUT) :: PAR(*)

      DOUBLE PRECISION, EXTERNAL :: GETP
      INTEGER NDX,NCOL,NTST

!     ---------------
!---------------------------------------------------------------------- 

!  Set PAR(3) equal to the minimum of U(2)
       PAR(9)=GETP('MIN',1,U)
! The following subroutines are not used here,
! but they must be supplied as dummy routines
      END SUBROUTINE PVLS
      SUBROUTINE BCND 
      END SUBROUTINE BCND

      SUBROUTINE ICND 
      END SUBROUTINE ICND

      SUBROUTINE FOPT 
      END SUBROUTINE FOPT



!----------------------------------------------------------------------
!----------------------------------------------------------------------
!---------------------------------------------------------------------- 
! END SUBROUTINE PVLS

!---------------------------------------------------------------------- 
! SUBROUTINE BCND
! END SUBROUTINE BCND

!---------------------------------------------------------------------- 
! SUBROUTINE ICND 
! END SUBROUTINE ICND

!----------------------------------------------------------------------
! SUBROUTINE FOPT 
! END SUBROUTINE FOPT

