from fenics import *
from random import sample
from math import floor
import numpy as np

parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True


'''
new set of tests including sparks as multiplicative random variable on mu 

and with a smaller active stress. We now replicate an oscillatory regime, but with the correct magnitudes of displacements. 

PLOT: for mu=0.3, lam = 0.1, and different values of T0 
BEST FIT: mu03_lam01_T0100



'''

mesh = Mesh("disk_small.xml")

#mu = Constant(0.288); lam = Constant(0.15); c0 = Constant(0.485044)
#mu = Constant(0.288); lam = Constant(0.1); c0 = Constant(0.485044)
mu = Constant(0.288); lam = Constant(0.5); c0 = Constant(0.682438894226448)
#mu = Constant(0.288); lam = Constant(1.3); c0 = Constant(0.9577649304341815)
#mu = Constant(0.3); lam = Constant(0.15); c0 = Constant(0.6038550678759639)
#---------------
#mu = Constant(0.3); lam = Constant(0.1); c0 = Constant(0.6038550678759639)
#---------------
#mu = Constant(0.3); lam = Constant(0.5); c0 = Constant(0.682438894226448)
#mu = Constant(0.3); lam = Constant(1.3); c0 = Constant(1.0136302384480085)

#T0      = Constant(100.) #Pa ZSeT
T0      = Constant(250.) #Pa ZNeW
#T0      = Constant(450.) #Pa  ZReW


fileO = XDMFFile(mesh.mpi_comm(), "outputs/ZREV-mu0288-lam05-T250-nu04-alphas0.xdmf")
outfile = open("mysol_mu0288_lam05_T250_nu04_alphas0.txt","w")

fileO.parameters['rewrite_function_mesh']=False
fileO.parameters["functions_share_mesh"] = True
fileO.parameters["flush_output"] = True

strain = lambda vec: sym(grad(vec))

# ********* Finite dimensional spaces ********* #

P1 = FiniteElement("CG", mesh.ufl_cell(), 1)
Bub  = FiniteElement("Bubble", mesh.ufl_cell(), 3)
P1b  = VectorElement(P1 + Bub)
RM   = VectorElement('R', triangle, 0, dim=3)


Eh = FunctionSpace(mesh,MixedElement([P1b,P1,RM]))
Rh = FunctionSpace(mesh,MixedElement([P1,P1]))

print("**************** total Dofs = ", Eh.dim() + Rh.dim())

c,h = TrialFunctions(Rh)
phi,psi = TestFunctions(Rh)
solR = Function(Rh)

ch = Function(Rh.sub(0).collapse())
uh = Function(Eh.sub(0).collapse())

solE  = Function(Eh)
u,p,s_chi = TrialFunctions(Eh)
v,q,s_xi  = TestFunctions(Eh)

nullspace=[Constant((1,0)), Constant((0,1)),\
               Expression(('-x[1]','x[0]'),degree = 1)]


# ********* model constants  ******* #

b      = Constant(0.111)
Dc     = Constant(20)     # mum^2/s
kf     = Constant(16.2)   # muM/s
gamma  = Constant(2.)     # muM/s
kgamma = Constant(0.1)    # muM
k1     = Constant(0.7)    # muM
tauj   = Constant(2.)     # s
L      = Constant(100.)   # mum 

D0 = Dc*tauj/L**2
K  = kgamma/k1
K1 = kf*tauj/k1
G  = gamma*tauj/k1 

alpha1_ = Constant(0.) #Constant(3790.) #Pa.s
alpha2_ = Constant(0.) #Constant(550.) #Pa.s




E0     = Constant(44.26)   #Pa
E1     = Constant(24.34)   #Pa
E = Expression('pow(x[0],2)+pow(x[1],2) <1.99? E0 : E1', E0=E0, E1=E1, degree = 0)
#nu0    = Constant(0.4)
#nu1    = Constant(0.3)
#nu = Expression('pow(x[0],2)+pow(x[1],2) <1.99? nu0 : nu1', nu0=nu0, nu1=nu1, degree = 0)
nu = Constant(0.4) #Constant(0.4)


alpha1 = alpha1_*(1+nu)/(E*tauj)
alpha2 = alpha2_*(1+nu)*(1-2*nu)/(E*nu*tauj)

beta1 = T0*(1+nu)/E
beta2 = Constant(1.0) #1/(k1**2)

# ********* Initial conditions ******* #

h0 = Constant(1.0/(1.0+pow(c0,2.0)))
cold = interpolate(c0,Rh.sub(0).collapse())
hold = interpolate(h0,Rh.sub(1).collapse())
uold = Function(Eh.sub(0).collapse())
pold = Function(Eh.sub(1).collapse())

# time constants
t=0.0; dt = 0.1; Tfinal = 30.2; freqSave = 2; inc = 0;

# multiplicative random perturbation of CICR with increasing magnitude in time

samples = 50
some_points = sample(list(mesh.coordinates()),samples)
ra = Constant(1.e4)
Ih = Expression('1+a*exp(-r*(pow(x[0]-q0,2)+pow(x[1]-q1,2)))', a=0, r=ra, q0=0, q1=0, degree=0)

amplj=[]; x0=[]; y0=[]

for j in range(samples):
    amplj.append(0.18+0.025*(j+1)**2/samples) # it was 20*(j+1)/samples. Now it is quadratic!! 
    x0.append(some_points[j][0])
    y0.append(some_points[j][1])
    
l = 0    

# ********* Weak forms ******* #

ELeft = (1+alpha1/dt)*inner(strain(u),strain(v)) * dx \
        + dot(s_chi,s_xi)*dx \
        - (1+alpha2/dt)* p * div(v) * dx \
        - (p*E*(1-2*nu)/(nu*(nu+1))+div(u))*q*dx

ERight = alpha1/dt * inner(strain(uold),strain(v)) * dx \
         - alpha2/dt * pold * div(v) * dx \
         - beta1*ch/(beta2+ch) * div(v) * dx # IT USED TO BE +!

for i, ns_i in enumerate(nullspace):
    chi = s_chi[i]
    xi  = s_xi[i]
    ELeft += chi*inner(v, ns_i)*dx + xi*inner(u, ns_i)*dx

RLeft = 1./dt*c*phi*dx + 1./dt*h*psi*dx \
        + D0*dot(grad(c),grad(phi)) * dx \
        + 1./dt*dot(uh-uold,grad(c)) * phi * dx

RRight = 1./dt*cold*phi*dx + 1./dt*hold*psi*dx \
         + (Ih*mu*hold*K1*(b+cold)/(1+cold)-G*cold/(K+cold))*phi*dx \
         + (1./(1.0+cold*cold)-hold) * psi * dx \
         + lam * div(uh) * phi * dx 
        
E_LHS = assemble(ELeft)
solverE = LUSolver(E_LHS, 'mumps')

volu = []; 

# ********* Time loop ************* # 
while (t <= Tfinal + dt):

    print("t=%.2f" % t)

    #INITIAL CONDITION:
    mag = conditional(lt(t,dt),0.1*c0,0.)
    ps = PointSource(Rh.sub(0), [(Point(0,0), mag)])

    E_RHS = assemble(ERight)
    
    solverE.solve(solE.vector(), E_RHS) 
    u,p,chi = solE.split()    
    assign(uh,u)

    #mesh1 = Mesh(mesh)
    #X = mesh1.coordinates()
    #X += np.vstack(map(u, X))
    #volume_after = assemble(Constant(1)*dx(domain=mesh1))
    volume_after = assemble(det(Identity(2) + grad(u))*dx)
    volu.append(volume_after)
    print(float(t+10), volu[inc], file = outfile)
    

    # print(' is this going down? ', int(Tfinal/(10*dt) + (50*dt**2-Tfinal)*t/(300*dt))+1.5)
    # SPARKS:  linear decay of frequency
    #y_a = Tfinal/(10*dt); y_b = 3*dt; t_a = 0; t_b = Tfinal;
    y_a = 1/0.075; y_b = 1/0.38; t_a = 0; t_b = Tfinal; 
    
    if (inc % int(y_a + (y_b-y_a)/(t_b-t_a)*(t-t_a)+1.5) == 0 and t>0.5):
        
        Ih.q0 = x0[l];
        Ih.q1 = y0[l];
        Ih.a  = amplj[l];
        
        print(' entering spark ', l, 'of amplitude',  float(amplj[l]) )
        l+=1
    
    R_LHS,R_RHS = assemble_system(RLeft,RRight)
    ps.apply(R_RHS)
    
    solve(R_LHS, solR.vector(), R_RHS, 'mumps')
    c,h = solR.split()
    assign(ch,c)
    
    if (inc % freqSave == 0):
        c.rename("c","c"); fileO.write(c,t)
        h.rename("h","h"); fileO.write(h,t)
        u.rename("u","u"); fileO.write(u,t)
        p.rename("p","p"); fileO.write(p,t)

    assign(uold,u); assign(pold,p);
    assign(cold,c); assign(hold,h)
    t += dt; inc += 1
    
# ************* End **************** #


'''
Katerina original data 

µ = 0.288 , λ = 0 , cStSt = 0.1777569513900799 : solitary wave
µ = 0.288 , λ = 0.1 , cStSt = 0.48504433614355585 : periodic wavetrain
µ = 0.288 , λ = 0.5 , cStSt = 0.682438894226448 : periodic wavetrain but slower speed
µ = 0.288 , λ = 1.3 , cStSt = 0.9577649304341815 : decaying wavetrain
µ = 0.288 , λ = 2 , cStSt = 1.1981692211685526 : very decaying wavetrain

µ = 0.3 , λ = 0 , cStSt = 0.5563278750155162 : periodic wavetrain
µ = 0.3 , λ = 0.1 , cStSt = 0.6038550678759639 : periodic wavetrain but slower speed
µ = 0.3 , λ = 0.5 , cStSt = 0.682438894226448 : periodic wavetrain but (even) slower speed
µ = 0.3 , λ = 1.3 , cStSt = 1.0136302384480085 : decaying wavetrain
µ = 0.3 , λ = 2 , cStSt = 1.2505584520106665 : very decaying wavetrain
µ = 0.5 , λ = 0 , cStSt = 1.3322905620049688 : decaying wavetrain

'''
